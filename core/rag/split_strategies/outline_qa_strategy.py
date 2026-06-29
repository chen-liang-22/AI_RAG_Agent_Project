"""PDF 目录问答切分策略。

适用于 PDF 书签目录里“一级章节 -> 二级问题”的资料。
用户选择该策略时，输出必须保留目录 metadata，不能静默降级成普通编号问答。
"""

from core.rag.split_strategies.base import BaseSplitStrategy, SplitContext


class OutlineQaSplitStrategy(BaseSplitStrategy):
    """PDF 目录问答切片策略。"""

    supported_split_strategies = ("outline_qa",)

    def split(self, context: SplitContext):
        """按 PDF 目录问答切片，失败时抛错提示用户改用其他切分策略。"""

        segments, qa_items = context.parser._build_outline_qa_segments(context.document_id, context.documents)
        if segments:
            return segments, qa_items
        raise ValueError(
            "目录问答切分失败：文件没有可用 PDF 目录，或目录标题无法在正文中定位。"
            "请改用编号问答切分、递归通用切分，或重新上传带有效书签目录的 PDF。"
        )
