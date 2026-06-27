"""V2 销售训练接口。"""

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import StreamingResponse

from api.schemas import DictionaryGroupResponse
from app_v2.application.training.goal_service import TrainingGoalApplicationService
from app_v2.application.training.material_service import TrainingMaterialApplicationService
from app_v2.application.training.plan_service import TrainingPlanApplicationService
from app_v2.application.training.profile_service import TrainingProfileApplicationService
from app_v2.application.training.scoring_service import TrainingScoringApplicationService
from app_v2.application.training.session_service import TrainingSessionApplicationService
from app_v2.application.training_support.schemas import (
    GoalSettingGenerateRequest,
    GoalSettingResponse,
    RoleGenerateRequest,
    RoleGenerateResponse,
    ScenarioPolishRequest,
    ScenarioPolishResponse,
    SupplementQuestionGenerateResponse,
    TrainingKnowledgeBatchListResponse,
    TrainingKnowledgeChunkListResponse,
    TrainingKnowledgeDeleteResponse,
    TrainingKnowledgePreviewResponse,
    TrainingKnowledgePublishResponse,
    TrainingKnowledgeReparseResponse,
    TrainingKnowledgeRollbackResponse,
    TrainingKnowledgeUploadResponse,
    TrainingKnowledgeVersionListResponse,
    TrainingPlanCreateRequest,
    TrainingPlanDeleteResponse,
    TrainingPlanDetailResponse,
    TrainingPlanListResponse,
    TrainingPlanUpdateRequest,
    TrainingScoreResponse,
    TrainingSessionDetailResponse,
    TrainingSessionListResponse,
    TrainingSessionResponse,
    TrainingSessionStartRequest,
    TrainingTurnRequest,
    TrainingTurnResponse,
)

router = APIRouter(prefix="/training", tags=["V2 销售训练"])


def _materials() -> TrainingMaterialApplicationService:
    return TrainingMaterialApplicationService()


def _profiles() -> TrainingProfileApplicationService:
    return TrainingProfileApplicationService()


def _goals() -> TrainingGoalApplicationService:
    return TrainingGoalApplicationService()


def _plans() -> TrainingPlanApplicationService:
    return TrainingPlanApplicationService()


def _sessions() -> TrainingSessionApplicationService:
    return TrainingSessionApplicationService()


def _scoring() -> TrainingScoringApplicationService:
    return TrainingScoringApplicationService()


@router.get("/profile-dictionaries", response_model=list[DictionaryGroupResponse])
def list_profile_dictionaries() -> list[DictionaryGroupResponse]:
    """查询销售训练画像字典。"""

    return _profiles().list_profile_dictionaries()


@router.post("/knowledge/upload", response_model=TrainingKnowledgeUploadResponse)
def upload_training_knowledge(
    file: UploadFile = File(...),
    source_type: str = Form("lms_case"),
    model_mode: str | None = Form(None),
    created_by: str | None = Form(None),
) -> TrainingKnowledgeUploadResponse:
    """上传销售训练知识，并生成待确认发布的预览切片。"""

    return _materials().upload(file=file, source_type=source_type, created_by=created_by, model_mode=model_mode)


@router.get("/knowledge/batches", response_model=TrainingKnowledgeBatchListResponse)
def list_training_batches(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=50)) -> TrainingKnowledgeBatchListResponse:
    """分页查询训练资料批次。"""

    return _materials().list_batches(page=page, page_size=page_size)


@router.get("/knowledge/batches/{batch_id}/preview", response_model=TrainingKnowledgePreviewResponse)
def preview_training_batch(batch_id: str, max_chars: int = Query(30000, ge=500, le=100000)) -> TrainingKnowledgePreviewResponse:
    """预览训练资料原文件内容。"""

    return _materials().preview_batch(batch_id, max_chars=max_chars)


@router.delete("/knowledge/batches/{batch_id}", response_model=TrainingKnowledgeDeleteResponse)
def delete_training_batch(batch_id: str) -> TrainingKnowledgeDeleteResponse:
    """删除训练资料批次。"""

    return _materials().delete_batch(batch_id)


@router.post("/knowledge/batches/{batch_id}/publish", response_model=TrainingKnowledgePublishResponse)
def publish_training_batch(batch_id: str) -> TrainingKnowledgePublishResponse:
    """人工确认发布训练资料。"""

    return _materials().publish_batch(batch_id)


@router.post("/knowledge/batches/{batch_id}/rollback", response_model=TrainingKnowledgeRollbackResponse)
def rollback_training_batch(batch_id: str) -> TrainingKnowledgeRollbackResponse:
    """回滚训练资料到指定历史版本。"""

    return _materials().rollback_batch(batch_id)


@router.post("/knowledge/batches/{batch_id}/reparse", response_model=TrainingKnowledgeReparseResponse)
def reparse_training_batch(
    batch_id: str,
    use_llm_fallback: bool = Query(True),
    model_mode: str | None = Query(None),
) -> TrainingKnowledgeReparseResponse:
    """重新切分未发布训练资料。"""

    return _materials().reparse_batch(batch_id, use_llm_fallback=use_llm_fallback, model_mode=model_mode)


@router.get("/knowledge/batches/{batch_id}/versions", response_model=TrainingKnowledgeVersionListResponse)
def list_training_batch_versions(batch_id: str) -> TrainingKnowledgeVersionListResponse:
    """查询训练资料版本链。"""

    return _materials().list_versions(batch_id)


@router.get("/knowledge/batches/{batch_id}/chunks", response_model=TrainingKnowledgeChunkListResponse)
def list_training_chunks(batch_id: str) -> TrainingKnowledgeChunkListResponse:
    """查询训练资料切片。"""

    return _materials().list_chunks(batch_id)


@router.post("/plans", response_model=TrainingPlanDetailResponse)
def create_training_plan(request: TrainingPlanCreateRequest) -> TrainingPlanDetailResponse:
    """创建训练方案。"""

    return _plans().create_plan(request)


@router.get("/plans", response_model=TrainingPlanListResponse)
def list_training_plans(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    keyword: str | None = None,
) -> TrainingPlanListResponse:
    """分页查询训练方案列表。"""

    return _plans().list_plans(page=page, page_size=page_size, keyword=keyword)


@router.get("/plans/{plan_id}", response_model=TrainingPlanDetailResponse)
def get_training_plan(plan_id: str) -> TrainingPlanDetailResponse:
    """查看训练方案每一步详情。"""

    return _plans().get_plan_detail(plan_id)


@router.delete("/plans/{plan_id}", response_model=TrainingPlanDeleteResponse)
def delete_training_plan(plan_id: str) -> TrainingPlanDeleteResponse:
    """删除训练方案。"""

    return _plans().delete_plan(plan_id)


@router.put("/plans/{plan_id}", response_model=TrainingPlanDetailResponse)
def update_training_plan(plan_id: str, request: TrainingPlanUpdateRequest) -> TrainingPlanDetailResponse:
    """修改训练方案某一步。"""

    return _plans().update_plan(plan_id, request)


@router.post("/profiles/generate", response_model=RoleGenerateResponse)
def generate_role_profile(request: RoleGenerateRequest) -> RoleGenerateResponse:
    """生成 AI 陪练角色。"""

    return _profiles().generate_role(request)


@router.post("/profiles/scenario/polish", response_model=ScenarioPolishResponse)
def polish_training_scenario(request: ScenarioPolishRequest) -> ScenarioPolishResponse:
    """根据客户画像字段润色训练场景描述。"""

    return _profiles().polish_scenario(request)


@router.post("/profiles/supplement-questions/generate", response_model=SupplementQuestionGenerateResponse)
def generate_role_supplement_questions(request: RoleGenerateRequest) -> SupplementQuestionGenerateResponse:
    """生成 AI 陪练角色前的补充问答题。"""

    return _profiles().generate_supplement_questions(request)


@router.post("/profiles/{profile_id}/goal-settings/generate", response_model=GoalSettingResponse)
def generate_goal_setting(profile_id: str, request: GoalSettingGenerateRequest) -> GoalSettingResponse:
    """生成一期开放式训练设置。"""

    return _goals().generate_goal_setting(profile_id, request)


@router.post("/sessions", response_model=TrainingSessionResponse)
def start_training_session(request: TrainingSessionStartRequest) -> TrainingSessionResponse:
    """开始训练会话。"""

    return _sessions().start_session(request)


@router.get("/sessions", response_model=TrainingSessionListResponse)
def list_training_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    trainee_id: str | None = None,
) -> TrainingSessionListResponse:
    """分页查询训练会话历史。"""

    return _sessions().list_sessions(page=page, page_size=page_size, trainee_id=trainee_id)


@router.get("/sessions/{session_id}", response_model=TrainingSessionDetailResponse)
def get_training_session_detail(session_id: str) -> TrainingSessionDetailResponse:
    """查询训练会话复盘详情。"""

    return _sessions().get_session_detail(session_id)


@router.post("/sessions/{session_id}/turns", response_model=TrainingTurnResponse)
def submit_training_turn(session_id: str, request: TrainingTurnRequest, stream: bool = Query(False)):
    """提交学员回复，支持一次性或流式返回。"""

    if stream or request.response_mode == "stream":
        return StreamingResponse(
            _sessions().stream_turn(session_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return _sessions().submit_turn(session_id, request)


@router.post("/sessions/{session_id}/final-score", response_model=TrainingScoreResponse)
def final_score(session_id: str, model_mode: str | None = None) -> TrainingScoreResponse:
    """结束训练并生成评分报告。"""

    return _scoring().final_score(session_id, model_mode=model_mode)
