"""V2 响应模型。"""

from pydantic import BaseModel, Field

from api.schemas import ConversationSummaryResponse, HealthResponse, KnowledgeFileResponse
from app_v2.application.training_support.schemas import TrainingKnowledgeBatchResponse, TrainingPlanSummaryResponse, TrainingSessionSummaryResponse


class DashboardOverviewResponse(BaseModel):
    """首页驾驶舱聚合响应。"""

    health: HealthResponse
    knowledge_files: list[KnowledgeFileResponse] = Field(default_factory=list)
    training_batches: list[TrainingKnowledgeBatchResponse] = Field(default_factory=list)
    training_plans: list[TrainingPlanSummaryResponse] = Field(default_factory=list)
    training_sessions: list[TrainingSessionSummaryResponse] = Field(default_factory=list)
    recent_conversations: list[ConversationSummaryResponse] = Field(default_factory=list)
    metrics: dict[str, int] = Field(default_factory=dict)
