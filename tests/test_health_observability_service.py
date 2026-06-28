"""可观测性健康检查服务测试。"""

from app.application.observability.health_service import (
    DependencyHealthCheck,
    DependencyHealthCheckResult,
    HealthDependencyService,
)


class FakeChecker(DependencyHealthCheck):
    """测试用依赖检查器。"""

    def __init__(self, result: DependencyHealthCheckResult):
        self.result = result
        self.called = False

    def check(self) -> DependencyHealthCheckResult:
        """返回预设检查结果。"""

        self.called = True
        return self.result


def test_dependency_health_service_marks_degraded_when_any_dependency_unavailable():
    """任一依赖不可用时，整体状态应降级，并保留每个依赖明细。"""

    mysql_checker = FakeChecker(DependencyHealthCheckResult(
        name="mysql",
        status="ok",
        latency_ms=12.5,
        message="MySQL 连接正常",
        details={"database": "ai_rag_agent"},
    ))
    redis_checker = FakeChecker(DependencyHealthCheckResult(
        name="redis",
        status="unavailable",
        latency_ms=2.0,
        message="Redis 未启用或不可用",
        details={"enabled": False},
    ))
    service = HealthDependencyService(checkers=[mysql_checker, redis_checker])

    response = service.check_dependencies()

    assert response.status == "degraded"
    assert response.summary == {"total": 2, "ok": 1, "unavailable": 1}
    assert [item.name for item in response.dependencies] == ["mysql", "redis"]
    assert response.dependencies[0].details["database"] == "ai_rag_agent"
    assert mysql_checker.called is True
    assert redis_checker.called is True
