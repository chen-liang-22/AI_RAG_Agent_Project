from rag.split_strategies.base import BaseSplitStrategy, SplitContext


class RecursiveSplitStrategy(BaseSplitStrategy):
    """普通文本递归切片策略。"""

    supported_split_strategies = ("recursive",)

    def split(self, context: SplitContext):
        """使用 RecursiveCharacterTextSplitter 切分普通文本。"""

        return context.parser._build_recursive_segments(
            context.document_id,
            context.documents,
            document_type=context.document_type,
        ), []
