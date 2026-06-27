import pytest

from utils import database_connection
from infrastructure import orm_session


def test_database_config_reads_mysql_pool_env(monkeypatch):
    """验证 MySQL ORM 连接池参数支持环境变量覆盖。"""

    monkeypatch.setenv("MYSQL_POOL_SIZE", "3")
    monkeypatch.setenv("MYSQL_MAX_OVERFLOW", "4")
    monkeypatch.setenv("MYSQL_POOL_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("MYSQL_POOL_RECYCLE_SECONDS", "900")

    config = database_connection.load_database_config()

    assert config["type"] == "mysql"
    assert config["mysql"]["pool_size"] == 3
    assert config["mysql"]["max_overflow"] == 4
    assert config["mysql"]["pool_timeout_seconds"] == 2.5
    assert config["mysql"]["pool_recycle_seconds"] == 900


def test_mysql_orm_engine_signature_includes_pool_options():
    """验证 ORM Engine 签名包含池参数，避免配置变更后继续复用旧 Engine。"""

    base_config = {
        "host": "127.0.0.1",
        "port": 3306,
        "database": "ai_rag_agent",
        "user": "root",
        "password": "secret",
        "charset": "utf8mb4",
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout_seconds": 5,
        "pool_recycle_seconds": 1800,
    }
    changed_config = dict(base_config)
    changed_config["pool_size"] = 8

    assert orm_session._mysql_engine_signature(base_config) != (
        orm_session._mysql_engine_signature(changed_config)
    )


def test_create_mysql_engine_returns_created_engine(monkeypatch):
    """创建 MySQL Engine 时必须返回绑定好的 Engine 对象，避免 Session 失去 bind。"""

    created = {"engine": object(), "url": None, "kwargs": None}

    def fake_create_engine(url, **kwargs):
        """记录 create_engine 入参，并返回测试 Engine 替身。"""

        created["url"] = url
        created["kwargs"] = kwargs
        return created["engine"]

    monkeypatch.setattr(orm_session, "create_engine", fake_create_engine)
    engine = orm_session._create_mysql_engine({
        "host": "127.0.0.1",
        "port": 3306,
        "database": "ai_rag_agent",
        "user": "root",
        "password": "secret",
        "charset": "utf8mb4",
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout_seconds": 5,
        "pool_recycle_seconds": 1800,
    })

    assert engine is created["engine"]
    assert created["kwargs"]["pool_pre_ping"] is True


def test_reset_orm_engines_disposes_existing_engines(monkeypatch):
    """验证重置 ORM Engine 缓存会释放旧 Engine，并清空 Session 工厂缓存。"""

    class FakeEngine:
        """测试用 Engine 替身，记录 dispose 是否被调用。"""

        def __init__(self):
            self.disposed = False

        def dispose(self):
            """模拟释放连接池中的连接。"""

            self.disposed = True

    fake_engine = FakeEngine()
    monkeypatch.setitem(orm_session._engine_cache, ("demo",), fake_engine)
    monkeypatch.setitem(orm_session._session_factory_cache, ("session",), object())

    orm_session.reset_orm_engines()

    assert fake_engine.disposed is True
    assert orm_session._engine_cache == {}
    assert orm_session._session_factory_cache == {}


def test_pytest_guard_blocks_real_business_database(monkeypatch):
    """pytest 环境不允许默认连接正式业务库，避免测试资料污染真实数据。"""

    monkeypatch.setenv("AI_RAG_TESTING", "1")
    monkeypatch.delenv("RUN_REAL_MYSQL_TESTS", raising=False)
    orm_session.clear_session_factory_override()
    monkeypatch.setattr(
        orm_session,
        "load_database_config",
        lambda: {
            "type": "mysql",
            "mysql": {
                "host": "127.0.0.1",
                "port": 3306,
                "database": "ai_rag_agent",
                "user": "root",
                "password": "secret",
                "charset": "utf8mb4",
                "pool_size": 5,
                "max_overflow": 10,
                "pool_timeout_seconds": 5,
                "pool_recycle_seconds": 1800,
            },
        },
    )

    with pytest.raises(RuntimeError, match="测试环境禁止连接正式 MySQL"):
        orm_session.get_orm_engine()


def test_session_factory_override_takes_precedence():
    """测试专用 SessionFactory 覆盖后，不再创建真实 MySQL Engine。"""

    fake_factory = object()

    try:
        orm_session.set_session_factory_override(fake_factory)

        assert orm_session.get_session_factory() is fake_factory
    finally:
        orm_session.clear_session_factory_override()
