from __future__ import annotations

import threading

import pytest

from infrastructure.id_generator import SnowflakeIdGenerator, new_id


def test_new_id_returns_numeric_string_with_safe_length():
    """new_id 应返回纯数字字符串，并且长度不超过现有 String(64) 字段。"""

    generated_id = new_id()

    assert generated_id.isdigit()
    assert len(generated_id) <= 64


def test_generator_creates_unique_ids_in_sequence():
    """连续生成大量 ID 时不能重复。"""

    generator = SnowflakeIdGenerator(worker_id=1)
    ids = [generator.next_id() for _ in range(10000)]

    assert len(ids) == len(set(ids))
    assert all(item.isdigit() for item in ids)


def test_generator_rejects_invalid_worker_id():
    """worker_id 超出 0-1023 范围时必须启动失败。"""

    with pytest.raises(ValueError, match="节点编号"):
        SnowflakeIdGenerator(worker_id=-1)

    with pytest.raises(ValueError, match="节点编号"):
        SnowflakeIdGenerator(worker_id=1024)


def test_generator_is_thread_safe():
    """多线程并发生成 ID 时不能重复。"""

    generator = SnowflakeIdGenerator(worker_id=2)
    generated_ids: list[str] = []
    lock = threading.Lock()

    def generate_many() -> None:
        local_ids = [generator.next_id() for _ in range(1000)]
        with lock:
            generated_ids.extend(local_ids)

    threads = [threading.Thread(target=generate_many) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(generated_ids) == 8000
    assert len(generated_ids) == len(set(generated_ids))


def test_generator_rejects_large_clock_backwards():
    """系统时间严重回拨时必须拒绝生成 ID。"""

    times = iter([2000, 1900])
    generator = SnowflakeIdGenerator(
        worker_id=3,
        max_clock_backward_ms=5,
        time_func=lambda: next(times),
        sleep_func=lambda milliseconds: None,
    )

    generator.next_id()
    with pytest.raises(RuntimeError, match="系统时间回拨"):
        generator.next_id()


def test_generator_waits_for_small_clock_backwards():
    """系统时间轻微回拨时等待后继续生成。"""

    times = iter([2000, 1998, 2001])
    slept: list[int] = []
    generator = SnowflakeIdGenerator(
        worker_id=4,
        max_clock_backward_ms=5,
        time_func=lambda: next(times),
        sleep_func=lambda milliseconds: slept.append(milliseconds),
    )

    first_id = generator.next_id()
    second_id = generator.next_id()

    assert first_id != second_id
    assert slept == [2]


def test_generator_waits_next_millisecond_when_sequence_overflows():
    """同一毫秒序列号耗尽时，应等待下一毫秒。"""

    current_times = [3000]
    slept: list[int] = []

    def time_func() -> int:
        return current_times[0]

    def sleep_func(milliseconds: int) -> None:
        slept.append(milliseconds)
        current_times[0] += 1

    generator = SnowflakeIdGenerator(worker_id=5, time_func=time_func, sleep_func=sleep_func)

    ids = [generator.next_id() for _ in range(4097)]

    assert len(ids) == len(set(ids))
    assert slept == [1]
