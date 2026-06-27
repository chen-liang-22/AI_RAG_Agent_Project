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
