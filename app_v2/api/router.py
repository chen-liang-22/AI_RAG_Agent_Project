"""V2 API 路由总入口。"""

from fastapi import APIRouter

from app_v2.api.routes import auth, chat, dashboard, dictionaries, exam, health, knowledge, training

router = APIRouter(prefix="/api/v2")

router.include_router(health.router)
router.include_router(auth.router)
router.include_router(dashboard.router)
router.include_router(dictionaries.router)
router.include_router(chat.router)
router.include_router(knowledge.router)
router.include_router(exam.router)
router.include_router(training.router)
