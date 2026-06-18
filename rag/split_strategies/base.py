from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class SplitContext:
    """知识库切片上下文。"""

    document_id: str
    documents: list[Document]
    document_type: str
    split_strategy: str
    parser: object


class BaseSplitStrategy(ABC):
    """知识库切片策略抽象基类。"""

    supported_split_strategies: tuple[str, ...] = ()

    def support_strategy(self, split_strategy: str) -> bool:
        """判断当前策略是否支持指定切分方式。"""

        return self.normalize_split_strategy(split_strategy) in self.supported_split_strategies

    @abstractmethod
    def split(self, context: SplitContext):
        """执行切片，并返回 segments 和 qa_items。"""

    @staticmethod
    def normalize_split_strategy(split_strategy: str) -> str:
        """归一化切分策略编码。"""

        return (split_strategy or "").strip().lower()
