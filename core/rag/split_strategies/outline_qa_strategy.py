"""PDF 目录问答切分策略。

适用于 PDF 书签目录里“一级章节 -> 二级问题”的资料。
优先使用 PDF outline 定位正文，失败后回退到编号问答切分。
"""

from core.rag.split_strategies.base import BaseSplitStrategy, SplitContext


class OutlineQaSplitStrategy(BaseSplitStrategy):
    """PDF 目录问答切片策略。"""

    supported_split_strategies = ("outline_qa",)

    def split(self, context: SplitContext):
        """优先按 PDF 目录问答切片，失败时回退到编号问答切片。"""

        segments, qa_items = context.parser._build_outline_qa_segments(context.document_id, context.documents)
        if segments:
            return segments, qa_items
        return context.parser._build_qa_segments(context.document_id, context.documents)
