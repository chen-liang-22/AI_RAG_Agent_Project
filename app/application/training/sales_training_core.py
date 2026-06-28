"""销售训练核心应用服务。

这个文件承载销售陪练一期的主业务编排：
- 训练资料上传、切片、质量评估、发布到正式向量库；
- 学员画像和客户画像合成 AI 客户角色；
- 生成开放式训练目标、动态轮数和评分规则；
- 训练会话对话、每轮检索案例证据、最终评分报告。

这里使用外观模式把多个子系统收敛成一个稳定入口。
文件较大是因为一期先保证流程闭环，后续可以继续按资料、角色、会话、评分拆小。
"""

from collections.abc import Iterator
from datetime import date, datetime

from fastapi import UploadFile

from app.infrastructure.repositories.document_repository import DocumentRepository
from app.infrastructure.vector_store_service import VectorStoreService
from app.application.training_support.repository import TrainingRepository
from app.application.training_support.schemas import (
    GoalSettingResponse,
    RoleGenerateRequest,
    RoleGenerateResponse,
    ScenarioPolishRequest,
    ScenarioPolishResponse,
    SupplementQuestionGenerateResponse,
    TrainingPlanCreateRequest,
    TrainingPlanDeleteResponse,
    TrainingPlanDetailResponse,
    TrainingPlanListResponse,
    TrainingPlanUpdateRequest,
    TrainingKnowledgeDeleteResponse,
    TrainingKnowledgeBatchListResponse,
    TrainingKnowledgePublishResponse,
    TrainingKnowledgeReparseResponse,
    TrainingKnowledgeRollbackResponse,
    TrainingKnowledgeVersionListResponse,
    TrainingSessionDetailResponse,
    TrainingSessionListResponse,
    TrainingKnowledgeChunkListResponse,
    TrainingKnowledgePreviewResponse,
    TrainingKnowledgeUploadResponse,
    TrainingScoreResponse,
    TrainingSessionResponse,
    TrainingSessionStartRequest,
    TrainingTurnRequest,
    TrainingTurnResponse,
)
from app.application.training.training_query_service import TrainingQueryService
from app.application.training.training_goal_application_service import TrainingGoalApplicationService
from app.application.training.training_goal_setting_service import TrainingGoalSettingService
from app.application.training.training_role_application_service import TrainingRoleApplicationService
from app.application.training.training_role_service import TrainingRoleService
from app.application.training.training_knowledge_service import TrainingKnowledgeService
from app.application.training.training_plan_domain_service import TrainingPlanDomainService
from app.application.training.training_session_basic_service import TrainingSessionBasicService
from app.application.training.training_session_prompt_service import TrainingSessionPromptService
from app.application.training.training_session_scoring_service import TrainingSessionScoringService
from app.application.training.training_session_turn_service import TrainingSessionTurnService
from core.utils.logger_handler import logger
from core.utils.config_handler import training_conf


DEFAULT_TRAINING_COLLECTION_NAME = "sales_training_cases"
DEFAULT_TRAINING_STAGING_COLLECTION_NAME = "sales_training_cases_staging"
TRAINING_COLLECTION_NAME = DEFAULT_TRAINING_COLLECTION_NAME


def _load_training_collection_config() -> dict[str, str]:
    """读取销售训练正式库和临时库 collection 配置。"""

    config = {
        "published": DEFAULT_TRAINING_COLLECTION_NAME,
        "staging": DEFAULT_TRAINING_STAGING_COLLECTION_NAME,
    }
    collection_config = training_conf.get("collections") if isinstance(training_conf, dict) else {}
    if not isinstance(collection_config, dict):
        return config

    published_collection = str(collection_config.get("published") or "").strip()
    staging_collection = str(collection_config.get("staging") or "").strip()
    if published_collection:
        config["published"] = published_collection
    if staging_collection:
        config["staging"] = staging_collection
    return config


def _format_response_time(value: object) -> str | None:
    """把数据库时间字段统一转换成接口响应字符串。"""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds", sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class V2SalesTrainingCoreService:
    """销售训练一期外观服务。

    外观模式用于把文件解析、向量库、LLM、业务数据库这些子系统收拢成
    前端能理解的训练流程接口。一期流程较短，暂不引入 Graph。
    """

    def __init__(
            self,
            repository: TrainingRepository | None = None,
            knowledge_store=None,
            document_repository: DocumentRepository | None = None,
    ):
        """初始化销售训练核心服务。

        这里组合训练仓储、文件台账、正式向量库和临时向量库。
        knowledge_store 只保留给旧测试兼容，真实文件台账统一走 DocumentRepository。
        """

        # repository 支持注入，主要是为了单元测试或局部替换仓储实现。
        self.repository = repository or TrainingRepository()
        # 文件台账复用知识库 documents 表，统一写入 MySQL。
        self.document_repository = document_repository or DocumentRepository(store=knowledge_store)
        collection_config = _load_training_collection_config()
        self.training_collection_name = collection_config["published"]
        self.staging_collection_name = collection_config["staging"]
        # 正式训练知识使用独立 collection，避免和智能客服的普通知识库混在一起。
        self.vector_service = VectorStoreService(collection_name=self.training_collection_name)
        # 待人工审核的上传切片写入临时 collection，发布成功后再清理，避免关系型数据库保存正文切片。
        self.staging_vector_service = VectorStoreService(collection_name=self.staging_collection_name)
        # 训练证据召回独立成查询服务，核心服务只负责业务编排。
        self.query_service = TrainingQueryService(
            repository=self.repository,
            vector_service=self.vector_service,
            collection_name=self.training_collection_name,
        )
        # 角色生成相关的纯逻辑拆到独立服务，核心外观只负责编排数据库、向量库和 LLM 调用。
        self.role_service = TrainingRoleService()
        # 角色生成应用流程拆到独立服务，核心外观只保留补充问题、场景润色和生成角色入口。
        self.role_application_service = TrainingRoleApplicationService(
            repository=self.repository,
            query_service=self.query_service,
            role_service=self.role_service,
        )
        # 训练目标生成同样拆成纯逻辑服务，避免核心编排类继续堆积提示词和兜底模板。
        self.goal_setting_service = TrainingGoalSettingService()
        # 训练目标生成应用流程拆到独立服务，核心外观只保留同名入口。
        self.goal_application_service = TrainingGoalApplicationService(
            repository=self.repository,
            goal_setting_service=self.goal_setting_service,
        )
        # 会话提示词和兜底话术拆到独立服务，核心外观继续负责仓库读写和流式编排。
        self.session_prompt_service = TrainingSessionPromptService()
        # 训练资料上传、预览、发布、回滚、重切和删除拆到独立服务，核心外观只保留稳定入口。
        self.knowledge_service = TrainingKnowledgeService(
            repository=self.repository,
            vector_service=self.vector_service,
            staging_vector_service=self.staging_vector_service,
            document_repository=self.document_repository,
            training_collection_name=self.training_collection_name,
            staging_collection_name=self.staging_collection_name,
        )
        # 训练方案 CRUD 和状态联动拆到方案领域服务，核心外观保留同名入口。
        self.plan_service = TrainingPlanDomainService(repository=self.repository)
        # 会话创建、历史列表和复盘详情拆到会话基础服务；对话流和评分后续单独拆。
        self.session_basic_service = TrainingSessionBasicService(
            repository=self.repository,
            session_prompt_service=self.session_prompt_service,
        )
        # 会话对话提交和 SSE 流式回复拆到会话对话服务；最终评分后续单独拆。
        self.session_turn_service = TrainingSessionTurnService(
            repository=self.repository,
            query_service=self.query_service,
            session_prompt_service=self.session_prompt_service,
        )
        # 最终评分拆到会话评分服务，核心服务只保留对外入口和跨模块编排。
        self.session_scoring_service = TrainingSessionScoringService(
            repository=self.repository,
            query_service=self.query_service,
            session_prompt_service=self.session_prompt_service,
        )
        logger.info(
            "[销售训练] 核心服务初始化完成 正式Collection=%s 临时Collection=%s",
            self.training_collection_name,
            self.staging_collection_name,
        )

    def upload_knowledge(
            self,
            *,
            file: UploadFile,
            source_type: str,
            created_by: str | None,
            model_mode: str | None = None,
    ) -> TrainingKnowledgeUploadResponse:
        """上传训练资料并生成待确认预览。

        主流程：
        1. 保存上传文件到 MinIO；
        2. 创建上传批次记录，状态为 parsing；
        3. 根据 source_type 选择切片策略；
        4. 对切片结果做质量评估；
        5. 保存切片明细；
        6. 状态改为 pending_review，等待人工确认发布。
        """

        return self.knowledge_service.upload_knowledge(
            file=file,
            source_type=source_type,
            created_by=created_by,
            model_mode=model_mode,
        )

    def list_batches(self, *, page: int = 1, page_size: int = 10) -> TrainingKnowledgeBatchListResponse:
        """分页查询已经上传过的训练资料。"""

        return self.knowledge_service.list_batches(page=page, page_size=page_size)

    def preview_batch(self, batch_id: str, *, max_chars: int = 30000) -> TrainingKnowledgePreviewResponse:
        """返回训练资料上传文件的站内预览数据。

        查看切片已经有独立入口，所以这里优先预览原文件文本；
        DOCX/PDF 会解析为文本，避免浏览器把 Word 文件当成下载处理。
        """

        return self.knowledge_service.preview_batch(batch_id, max_chars=max_chars)

    def delete_batch(self, batch_id: str) -> TrainingKnowledgeDeleteResponse:
        """删除训练资料批次，并通过统一文件资产服务清理全链路数据。"""

        return self.knowledge_service.delete_batch(batch_id)

    def _delete_legacy_batch_without_document(self, batch_id: str) -> TrainingKnowledgeDeleteResponse:
        """删除没有 document_id 的历史训练批次。

        老数据只存在 training_knowledge_batches 和训练向量库里，无法走 documents 统一文件资产链路。
        因此这里按 batch_id 清理正式库、临时库和批次记录，保留历史数据兼容能力。
        """

        return self.knowledge_service.delete_legacy_batch_without_document(batch_id)

    def publish_batch(self, batch_id: str) -> TrainingKnowledgePublishResponse:
        """人工确认发布训练资料。

        上传阶段已经把待审核切片写入临时 Qdrant collection。
        发布阶段只把临时向量点复制到正式 collection，成功后删除临时点。
        """

        return self.knowledge_service.publish_batch(batch_id)

    def rollback_batch(self, batch_id: str) -> TrainingKnowledgeRollbackResponse:
        """回滚训练资料到指定历史版本。

        历史版本的正式向量点会长期保留，回滚时只切换当前版本标记和业务数据库状态。
        """

        return self.knowledge_service.rollback_batch(batch_id)

    def reparse_batch(
            self,
            batch_id: str,
            *,
            use_llm_fallback: bool = True,
            model_mode: str | None = None,
    ) -> TrainingKnowledgeReparseResponse:
        """重新切分未发布训练资料。

        该接口用于人工预览发现规则切分不理想时，主动触发 LLM 兜底切分。
        已发布版本不能直接重切，避免绕过人工确认并破坏临时库到正式库的发布边界。
        """

        return self.knowledge_service.reparse_batch(
            batch_id,
            use_llm_fallback=use_llm_fallback,
            model_mode=model_mode,
        )

    def list_batch_versions(self, batch_id: str) -> TrainingKnowledgeVersionListResponse:
        """查询指定训练资料所在版本组的版本链。"""

        return self.knowledge_service.list_batch_versions(batch_id)

    def list_chunks(self, batch_id: str) -> TrainingKnowledgeChunkListResponse:
        """查询某个上传批次的训练知识切片。"""

        return self.knowledge_service.list_chunks(batch_id)

    def create_plan(self, request: TrainingPlanCreateRequest) -> TrainingPlanDetailResponse:
        """创建训练方案。

        训练方案先保存“输入快照”，还不会自动生成角色。
        这样用户可以先命名、检查画像和场景，再按步骤生成后续内容。
        """

        return self.plan_service.create_plan(request)

    def list_plans(self, *, page: int = 1, page_size: int = 10, keyword: str | None = None) -> TrainingPlanListResponse:
        """分页查询训练方案列表。"""

        return self.plan_service.list_plans(page=page, page_size=page_size, keyword=keyword)

    def get_plan_detail(self, plan_id: str) -> TrainingPlanDetailResponse:
        """查询训练方案完整详情。"""

        return self.plan_service.get_plan_detail(plan_id)

    def delete_plan(self, plan_id: str) -> TrainingPlanDeleteResponse:
        """删除训练方案。

        训练方案是销售陪练配置入口。删除它只会让方案从列表和详情里消失，
        不清理训练资料、向量库、MinIO 文件，也不删除历史训练会话依赖的角色和阶段配置。
        """

        return self.plan_service.delete_plan(plan_id)

    def update_plan(self, plan_id: str, request: TrainingPlanUpdateRequest) -> TrainingPlanDetailResponse:
        """修改训练方案。

        依赖关系说明：
        - 修改学员画像/客户画像/场景/补充细节：角色已不可信，阶段和评分也必须重新生成；
        - 修改角色扮演画像或隐藏画像：训练阶段和评分必须重新生成；
        - 修改训练阶段：评分规则需要重新确认；
        - 只修改评分规则：不影响前面的角色和阶段。
        """

        return self.plan_service.update_plan(plan_id, request)

    def generate_supplement_questions(self, request: RoleGenerateRequest) -> SupplementQuestionGenerateResponse:
        """生成 AI 陪练角色前的补充问答题。

        这是角色生成的前置澄清步骤：先让管理员选择客户真实顾虑、价值判断、
        业务痛点等细节，再把答案并入 extra_details 生成更稳定的角色。
        """

        return self.role_application_service.generate_supplement_questions(request)

    def polish_scenario(self, request: ScenarioPolishRequest) -> ScenarioPolishResponse:
        """根据客户画像字段润色训练场景描述。

        这是销售陪练服务的一个小外观方法：前端只关心“把场景润色好”，
        具体调用哪个模型、如何兜底，都收敛在服务层。
        """

        return self.role_application_service.polish_scenario(request)

    def generate_role(self, request: RoleGenerateRequest) -> RoleGenerateResponse:
        """生成 AI 陪练角色。

        角色生成不是单纯让 LLM 编故事，而是先从训练向量库召回案例证据，
        再把“学员画像 + 客户字段 + 场景 + 证据”一起交给模型。
        """

        return self.role_application_service.generate_role(request)

    def generate_goal_setting(
            self,
            *,
            profile_id: str,
            trainee_id: str,
            training_mode: str,
            plan_id: str | None = None,
            model_mode: str | None = None,
    ) -> GoalSettingResponse:
        """生成一期开放式训练设置。

        一期只支持开放式训练，所以这里只生成一个阶段。
        round_limit 由 LLM 根据角色复杂度动态给出，后端再做 5-100 的安全边界。
        """

        return self.goal_application_service.generate_goal_setting(
            profile_id=profile_id,
            trainee_id=trainee_id,
            training_mode=training_mode,
            plan_id=plan_id,
            model_mode=model_mode,
        )

    def start_session(self, request: TrainingSessionStartRequest) -> TrainingSessionResponse:
        """开始一次开放式训练。

        和普通聊天不同，训练一开始应该由 AI 客户先“在场”。
        所以创建会话后会立刻生成并保存 round_no=0 的客户开场白。
        """

        return self.session_basic_service.start_session(request)

    def list_sessions(
            self,
            *,
            page: int = 1,
            page_size: int = 10,
            trainee_id: str | None = None,
    ) -> TrainingSessionListResponse:
        """分页查询训练历史。"""

        return self.session_basic_service.list_sessions(page=page, page_size=page_size, trainee_id=trainee_id)

    def get_session_detail(self, session_id: str) -> TrainingSessionDetailResponse:
        """查询训练复盘详情。

        这个接口给前端“最近训练”使用：
        - session：会话摘要；
        - turns：完整对话；
        - role_profile：角色确认卡片；
        - goal_setting：训练目标；
        - score：已有评分报告。
        """

        return self.session_basic_service.get_session_detail(session_id)

    def submit_turn(self, session_id: str, request: TrainingTurnRequest) -> TrainingTurnResponse:
        """提交学员回复并一次性返回 AI 客户回复。"""

        return self.session_turn_service.submit_turn(session_id, request)

    def stream_turn(self, session_id: str, request: TrainingTurnRequest) -> Iterator[str]:
        """提交学员回复并返回 SSE 流。

        Python 的 Iterator[str] + yield 是生成器写法。
        FastAPI StreamingResponse 会一边读取 yield 出来的字符串，一边推给浏览器。

        这里返回的是 SSE 文本事件：
        - retrieval_done：本轮检索完成；
        - customer_delta：AI 客户回复增量；
        - stage_decision：阶段/会话状态；
        - turn_done：本轮完成；
        - error：异常。
        """

        yield from self.session_turn_service.stream_turn(session_id, request)

    def final_score(self, session_id: str, model_mode: str | None = None) -> TrainingScoreResponse:
        """结束训练并生成评分报告。"""

        return self.session_scoring_service.final_score(session_id, model_mode=model_mode)

