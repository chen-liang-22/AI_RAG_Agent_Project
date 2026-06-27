from core.rag.split_strategies.base import BaseSplitStrategy, SplitContext


class NumberedQaSplitStrategy(BaseSplitStrategy):
    """编号问答切片策略。"""

    supported_split_strategies = ("numbered_qa",)

    def split(self, context: SplitContext):
        """把编号问答文档切成 QA segment。"""

        return context.parser._build_qa_segments(context.document_id, context.documents)
