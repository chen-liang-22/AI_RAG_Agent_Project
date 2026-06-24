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
