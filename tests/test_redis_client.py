from types import SimpleNamespace

from utils import redis_client
from utils.redis_client import RedisClient


class FakeRedisLock:
    """测试用 Redis 锁替身，记录是否被释放。"""

    def __init__(self):
        self.released = False

    def acquire(self, blocking=True):
        """模拟立即获取锁成功。"""

        return True

    def release(self):
        """模拟释放锁。"""

        self.released = True


class FakeRedisConnection:
    """测试用 Redis 连接替身，避免单元测试依赖真实 Redis 服务。"""

    def __init__(self):
        self.values: dict[str, str] = {}
        self.last_lock = FakeRedisLock()

    def ping(self):
        """模拟 Redis PING 成功。"""

        return True

    def get(self, key):
        """模拟读取字符串值。"""

        return self.values.get(key)

    def set(self, name, value, ex=None):
        """模拟写入字符串值。"""

        self.values[name] = value
        return True

    def delete(self, *keys):
        """模拟删除多个 key。"""

        count = 0
        for key in keys:
            if key in self.values:
                count += 1
                del self.values[key]
        return count

    def lock(self, name, timeout=None, blocking_timeout=None):
        """模拟创建 Redis 分布式锁。"""

        self.last_lock = FakeRedisLock()
        return self.last_lock


def test_redis_client_degrades_when_disabled():
    client = RedisClient({"enabled": False, "key_prefix": "test"})

    assert client.is_available() is False
    assert client.get_json("missing", default={"fallback": True}) == {"fallback": True}
    assert client.set_json("anything", {"ok": True}) is False
    assert client.delete("anything") == 0


def test_redis_client_json_task_status_and_lock(monkeypatch):
    fake_connection = FakeRedisConnection()
    monkeypatch.setattr(redis_client, "redis", SimpleNamespace(Redis=lambda **kwargs: fake_connection))

    client = RedisClient({
        "enabled": True,
        "key_prefix": "test_app",
        "default_ttl_seconds": 60,
        "lock_timeout_seconds": 10,
    })
    key = client.build_key("cache", "demo")

    assert client.is_available() is True
    assert key == "test_app:cache:demo"
    assert client.set_json(key, {"name": "Redis"}) is True
    assert client.get_json(key) == {"name": "Redis"}
    assert client.set_task_status("upload", "batch_001", "running", {"progress": 50}) is True
    assert client.get_task_status("upload", "batch_001") == {
        "task_type": "upload",
        "task_id": "batch_001",
        "status": "running",
        "progress": 50,
    }

    with client.lock("publish", "batch_001") as acquired:
        assert acquired is True

    assert fake_connection.last_lock.released is True
    assert client.delete(key) == 1
    assert client.get_json(key) is None
