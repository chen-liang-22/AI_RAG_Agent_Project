"""销售训练画像应用服务。"""

from api.schemas import DictionaryGroupResponse
from app_v2.application.dictionary_service import DictionaryApplicationService
from app_v2.infrastructure.repositories.dictionary_repository import DictionaryRepository
from training.schemas import (
    RoleGenerateRequest,
    RoleGenerateResponse,
    ScenarioPolishRequest,
    ScenarioPolishResponse,
    SupplementQuestionGenerateResponse,
)
from utils.logger_handler import logger

from .service_provider import get_training_core_service

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


class TrainingProfileApplicationService:
    """训练画像外观服务。"""

    def __init__(self, service=None, store=None, dictionary_repository: DictionaryRepository | None = None, core_service=None):
        self._core_service = core_service or service
        self.service = None
        # store 是旧构造参数，保留在函数签名里只为兼容历史调用方。
        # 新流程统一走 V2 字典仓储，避免画像服务继续直连旧 KnowledgeStore。
        self.dictionary_repository = dictionary_repository or DictionaryRepository()

    @property
    def core_service(self):
        """延迟获取 V2 销售训练核心服务，画像字典查询不触发它。"""

        if self._core_service is None:
            self._core_service = get_training_core_service()
        return self._core_service

    def list_profile_dictionaries(self) -> list[DictionaryGroupResponse]:
        """查询销售训练画像相关字典。"""

        rows = []
        for dictionary_code in PROFILE_DICTIONARY_CODES:
            rows.extend(self.dictionary_repository.list_items(dictionary_code=dictionary_code))
        return DictionaryApplicationService._build_dictionary_groups(rows)

    def generate_role(self, request: RoleGenerateRequest) -> RoleGenerateResponse:
        """生成 AI 陪练角色。"""

        logger.info("[V2销售训练-画像] 生成陪练角色开始 训练名称=%s", getattr(request, "plan_name", None))
        return self.core_service.generate_role(request)

    def polish_scenario(self, request: ScenarioPolishRequest) -> ScenarioPolishResponse:
        """根据客户画像润色训练场景。"""

        return self.core_service.polish_scenario(request)

    def generate_supplement_questions(self, request: RoleGenerateRequest) -> SupplementQuestionGenerateResponse:
        """生成角色补充问答。"""

        return self.core_service.generate_supplement_questions(request)
