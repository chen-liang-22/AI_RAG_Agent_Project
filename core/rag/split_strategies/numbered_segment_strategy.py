"""编号条目切分策略。

适用于只有编号条目、但不一定是标准问答的资料。
如果没有识别出编号条目，会自动回退到普通递归切分。
"""

from core.rag.split_strategies.base import BaseSplitStrategy, SplitContext


class NumberedSegmentSplitStrategy(BaseSplitStrategy):
    """编号条目切片策略。"""

    supported_split_strategies = ("numbered_segments",)

    def split(self, context: SplitContext):
        """按编号条目切片，无法切出内容时回退到递归切片。"""

        segments = context.parser._build_numbered_segments(
            context.document_id,
            context.documents,
            document_type=context.document_type,
        )
        if segments:
            return segments, []
        return context.parser._build_recursive_segments(
            context.document_id,
            context.documents,
            document_type=context.document_type,
        ), []
