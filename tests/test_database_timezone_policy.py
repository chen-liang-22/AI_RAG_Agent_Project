"""数据库业务时间时区策略测试。"""

from datetime import datetime, timezone

from app.application.training_support import repository


class FixedDateTime(datetime):
    """固定当前 UTC 时间，便于验证东八区转换。"""

    @classmethod
    def now(cls, tz=None):
        """返回固定时间，模拟 UTC 2026-07-01 00:30:00。"""

        base_time = datetime(2026, 7, 1, 0, 30, 0, tzinfo=timezone.utc)
        if tz is not None:
            return base_time.astimezone(tz)
        return base_time.replace(tzinfo=None)


def test_database_now_uses_east_8_naive_datetime(monkeypatch):
    """业务数据库时间应按东八区保存，并去掉时区信息写入 DATETIME。"""

    monkeypatch.setattr(repository, "datetime", FixedDateTime)

    now = repository.utc_now()

    assert now == datetime(2026, 7, 1, 8, 30, 0)
    assert now.tzinfo is None


def test_database_now_text_uses_east_8_mysql_text(monkeypatch):
    """业务数据库时间文本应保持东八区的 MySQL DATETIME 格式。"""

    monkeypatch.setattr(repository, "datetime", FixedDateTime)

    assert repository.utc_now_text() == "2026-07-01 08:30:00"
