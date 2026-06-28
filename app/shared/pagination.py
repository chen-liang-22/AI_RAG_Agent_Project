"""分页和模糊查询工具。"""

from dataclasses import dataclass

from app.domain.constants import DEFAULT_PAGE, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE


@dataclass(frozen=True)
class PageRequest:
    """标准分页请求。

    Repository 只关心 page/page_size/offset，不再到处手写分页计算。
    """

    page: int
    page_size: int

    @property
    def offset(self) -> int:
        """MySQL 查询使用的起始偏移量。"""

        return (self.page - 1) * self.page_size


def normalize_page(page: int | None, page_size: int | None, *, max_page_size: int = MAX_PAGE_SIZE) -> PageRequest:
    """把外部传入的分页参数收敛到安全范围。"""

    safe_page = max(DEFAULT_PAGE, int(page or DEFAULT_PAGE))
    safe_page_size = max(1, min(int(page_size or DEFAULT_PAGE_SIZE), max_page_size))
    return PageRequest(page=safe_page, page_size=safe_page_size)


def escape_like_keyword(keyword: str) -> str:
    """转义 SQL LIKE 关键字中的特殊字符。

    `%` 和 `_` 在 LIKE 中有通配含义，直接拼进去会导致查询范围扩大。
    """

    return keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
