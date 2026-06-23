from utils import database_connection


def test_database_config_reads_mysql_pool_env(monkeypatch):
    """验证 MySQL 连接池参数支持环境变量覆盖。"""

    monkeypatch.setenv("DATABASE_TYPE", "mysql")
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


def test_mysql_pool_signature_includes_pool_options():
    """验证连接池签名包含池参数，避免配置变更后继续复用旧连接池。"""

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

    assert database_connection._mysql_pool_config_signature(base_config) != (
        database_connection._mysql_pool_config_signature(changed_config)
    )


def test_get_mysql_pool_reuses_pool_without_opening_connection(monkeypatch):
    """验证相同配置会复用同一个连接池，且创建连接池时不会立刻连接 MySQL。"""

    database_connection.reset_mysql_pool()

    def forbidden_creator(config):
        """如果创建池阶段真的连接 MySQL，测试应立即失败。"""

        raise AssertionError("创建连接池时不应该立刻连接 MySQL")

    monkeypatch.setattr(database_connection, "_create_raw_mysql_connection", forbidden_creator)
    config = {
        "host": "127.0.0.1",
        "port": 3306,
        "database": "ai_rag_agent",
        "user": "root",
        "password": "secret",
        "charset": "utf8mb4",
        "pool_size": 2,
        "max_overflow": 1,
        "pool_timeout_seconds": 1,
        "pool_recycle_seconds": 60,
    }

    first_pool = database_connection._get_mysql_pool(config)
    second_pool = database_connection._get_mysql_pool(config)

    assert first_pool is second_pool
    assert first_pool.size() == 2
    database_connection.reset_mysql_pool()


def test_reset_mysql_pool_disposes_existing_pool(monkeypatch):
    """验证重置连接池会释放旧池，并清空缓存签名。"""

    class FakePool:
        """测试用连接池替身，记录 dispose 是否被调用。"""

        def __init__(self):
            self.disposed = False

        def dispose(self):
            """模拟释放连接池中的连接。"""

            self.disposed = True

    fake_pool = FakePool()
    monkeypatch.setattr(database_connection, "_mysql_pool", fake_pool)
    monkeypatch.setattr(database_connection, "_mysql_pool_signature", ("demo",))

    database_connection.reset_mysql_pool()

    assert fake_pool.disposed is True
    assert database_connection._mysql_pool is None
    assert database_connection._mysql_pool_signature is None
