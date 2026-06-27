"""文档切分策略工厂。

这里使用工厂方法模式，根据 split_strategy 编码返回对应切分策略。
如果配置或前端传入未知策略，会回退到递归切分，保证上传流程不中断。
"""

from core.rag.split_strategies.base import BaseSplitStrategy
from core.rag.split_strategies.llm_semantic_strategy import LlmSemanticSplitStrategy
from core.rag.split_strategies.numbered_qa_strategy import NumberedQaSplitStrategy
from core.rag.split_strategies.numbered_segment_strategy import NumberedSegmentSplitStrategy
from core.rag.split_strategies.outline_qa_strategy import OutlineQaSplitStrategy
from core.rag.split_strategies.recursive_strategy import RecursiveSplitStrategy


class SplitStrategyFactory:
    """知识库切片策略工厂。"""

    _strategies: list[BaseSplitStrategy] = []

    @classmethod
    def ensure_default_strategies(cls) -> None:
        """注册系统默认切片策略。"""

        if cls._strategies:
            return
        cls.register(LlmSemanticSplitStrategy())
        cls.register(OutlineQaSplitStrategy())
        cls.register(NumberedQaSplitStrategy())
        cls.register(NumberedSegmentSplitStrategy())
        cls.register(RecursiveSplitStrategy())

    @classmethod
    def register(cls, strategy: BaseSplitStrategy) -> None:
        """注册一个切片策略。"""

        strategy_class = strategy.__class__
        if any(isinstance(existing, strategy_class) for existing in cls._strategies):
            return
        cls._strategies.append(strategy)

    @classmethod
    def get_strategy(cls, split_strategy: str) -> BaseSplitStrategy:
        """按切分策略编码获取对应策略。"""

        cls.ensure_default_strategies()
        normalized_strategy = BaseSplitStrategy.normalize_split_strategy(split_strategy)
        for strategy in cls._strategies:
            if strategy.support_strategy(normalized_strategy):
                return strategy
        return RecursiveSplitStrategy()
