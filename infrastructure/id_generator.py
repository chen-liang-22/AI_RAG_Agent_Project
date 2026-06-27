"""雪花算法主键生成器。

本模块只负责生成数据库业务主键。返回值统一是纯数字字符串，
历史带前缀 ID 继续由现有数据保留，不在这里处理。
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import yaml

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


DEFAULT_EPOCH = "2026-01-01T00:00:00+08:00"
DEFAULT_WORKER_ID = 0
DEFAULT_MAX_CLOCK_BACKWARD_MS = 5
WORKER_ID_BITS = 10
SEQUENCE_BITS = 12
MAX_WORKER_ID = (1 << WORKER_ID_BITS) - 1
MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1
WORKER_ID_SHIFT = SEQUENCE_BITS
TIMESTAMP_SHIFT = WORKER_ID_BITS + SEQUENCE_BITS


@dataclass(frozen=True)
class IdGeneratorConfig:
    """雪花算法配置。"""

    enabled: bool
    epoch: str
    epoch_ms: int
    worker_id: int
    max_clock_backward_ms: int


class SnowflakeIdGenerator:
    """线程安全的雪花 ID 生成器。"""

    def __init__(
            self,
            *,
            worker_id: int,
            epoch_ms: int = 0,
            max_clock_backward_ms: int = DEFAULT_MAX_CLOCK_BACKWARD_MS,
            time_func: Callable[[], int] | None = None,
            sleep_func: Callable[[int], None] | None = None,
    ):
        """初始化生成器并校验节点编号。"""

        if worker_id < 0 or worker_id > MAX_WORKER_ID:
            raise ValueError(f"节点编号必须在 0-{MAX_WORKER_ID} 之间，当前值={worker_id}")
        if max_clock_backward_ms < 0:
            raise ValueError(f"时钟回拨容忍毫秒不能小于 0，当前值={max_clock_backward_ms}")

        self.worker_id = worker_id
        self.epoch_ms = epoch_ms
        self.max_clock_backward_ms = max_clock_backward_ms
        self.time_func = time_func or _current_time_ms
        self.sleep_func = sleep_func or _sleep_ms
        self._lock = threading.Lock()
        self._last_timestamp = -1
        self._sequence = 0

    def next_id(self) -> str:
        """生成纯数字字符串 ID。"""

        with self._lock:
            timestamp = self._next_valid_timestamp()
            if timestamp == self._last_timestamp:
                self._sequence = (self._sequence + 1) & MAX_SEQUENCE
                if self._sequence == 0:
                    timestamp = self._wait_next_millisecond(timestamp)
            else:
                self._sequence = 0

            self._last_timestamp = timestamp
            snowflake_id = (
                ((timestamp - self.epoch_ms) << TIMESTAMP_SHIFT)
                | (self.worker_id << WORKER_ID_SHIFT)
                | self._sequence
            )
            return str(snowflake_id)

    def _next_valid_timestamp(self) -> int:
        """获取可用毫秒时间，处理系统时间回拨。"""

        timestamp = self.time_func()
        if timestamp >= self._last_timestamp:
            return timestamp

        backward_ms = self._last_timestamp - timestamp
        if backward_ms > self.max_clock_backward_ms:
            logger.error(
                "[主键生成] 系统时间回拨超过阈值 上次毫秒=%s 当前毫秒=%s 回拨毫秒=%s",
                self._last_timestamp,
                timestamp,
                backward_ms,
            )
            raise RuntimeError(f"系统时间回拨超过阈值，回拨毫秒={backward_ms}")

        self.sleep_func(backward_ms)
        retry_timestamp = self.time_func()
        if retry_timestamp < self._last_timestamp:
            raise RuntimeError(f"系统时间回拨后仍不可用，回拨毫秒={self._last_timestamp - retry_timestamp}")
        return retry_timestamp

    def _wait_next_millisecond(self, current_timestamp: int) -> int:
        """等待进入下一毫秒，避免同毫秒序列号溢出。"""

        next_timestamp = self.time_func()
        while next_timestamp <= current_timestamp:
            self.sleep_func(1)
            next_timestamp = self.time_func()
        return next_timestamp


def new_id() -> str:
    """生成数据库业务主键，返回纯雪花数字字符串。"""

    return get_id_generator().next_id()


def get_id_generator() -> SnowflakeIdGenerator:
    """返回进程内单例雪花 ID 生成器。"""

    global _generator
    if _generator is None:
        with _generator_lock:
            if _generator is None:
                config = load_id_generator_config()
                _generator = SnowflakeIdGenerator(
                    worker_id=config.worker_id,
                    epoch_ms=config.epoch_ms,
                    max_clock_backward_ms=config.max_clock_backward_ms,
                )
                logger.info(
                    "[主键生成] 雪花算法配置加载完成 节点编号=%s 起始时间=%s",
                    config.worker_id,
                    config.epoch,
                )
    return _generator


def reset_id_generator_for_test() -> None:
    """重置单例生成器，仅供测试隔离使用。"""

    global _generator
    with _generator_lock:
        _generator = None


def load_id_generator_config(config_path: str | None = None) -> IdGeneratorConfig:
    """读取雪花算法配置，并应用环境变量覆盖。"""

    path = config_path or get_abs_path("config/id_generator.yml")
    raw_config = _read_yaml(path)
    epoch = str(os.getenv("ID_GENERATOR_EPOCH") or raw_config.get("epoch") or DEFAULT_EPOCH)
    worker_id = _read_int("ID_GENERATOR_WORKER_ID", raw_config.get("worker_id"), DEFAULT_WORKER_ID)
    max_clock_backward_ms = _read_int(
        "ID_GENERATOR_MAX_CLOCK_BACKWARD_MS",
        raw_config.get("max_clock_backward_ms"),
        DEFAULT_MAX_CLOCK_BACKWARD_MS,
    )
    enabled = _read_bool("ID_GENERATOR_ENABLED", raw_config.get("enabled"), True)
    return IdGeneratorConfig(
        enabled=enabled,
        epoch=epoch,
        epoch_ms=_parse_epoch_ms(epoch),
        worker_id=worker_id,
        max_clock_backward_ms=max_clock_backward_ms,
    )


def _read_yaml(path: str) -> dict[str, Any]:
    """读取 YAML 配置文件，不存在时返回空配置。"""

    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def _read_int(env_name: str, raw_value: Any, default: int) -> int:
    """读取整数配置，环境变量优先。"""

    value = os.getenv(env_name)
    source = value if value not in (None, "") else raw_value
    try:
        return int(source)
    except (TypeError, ValueError):
        logger.warning("[主键生成] 整数配置非法 配置项=%s 值=%s，使用默认值=%s", env_name, source, default)
        return default


def _read_bool(env_name: str, raw_value: Any, default: bool) -> bool:
    """读取布尔配置，环境变量优先。"""

    value = os.getenv(env_name)
    source = value if value not in (None, "") else raw_value
    if source in (None, ""):
        return default
    if isinstance(source, bool):
        return source
    return str(source).strip().lower() in {"1", "true", "yes", "on"}


def _parse_epoch_ms(epoch: str) -> int:
    """把 ISO 起始时间转换成毫秒时间戳。"""

    try:
        return int(datetime.fromisoformat(epoch).timestamp() * 1000)
    except ValueError as exc:
        raise ValueError(f"雪花算法起始时间格式不正确：{epoch}") from exc


def _current_time_ms() -> int:
    """返回当前毫秒时间戳。"""

    return int(time.time() * 1000)


def _sleep_ms(milliseconds: int) -> None:
    """按毫秒休眠。"""

    time.sleep(milliseconds / 1000)


_generator: SnowflakeIdGenerator | None = None
_generator_lock = threading.Lock()
