"""编号问答切分策略。

适用于“1. 问题 / 答：答案”这类明确编号问答资料。
具体编号和答案前缀规则来自 config/app.yml 的 rag.document_parse_rules。
"""

from core.rag.split_strategies.base import BaseSplitStrategy, SplitContext


class NumberedQaSplitStrategy(BaseSplitStrategy):
    """编号问答切片策略。"""

    supported_split_strategies = ("numbered_qa",)

    def split(self, context: SplitContext):
        """把编号问答文档切成 QA segment。"""

        return context.parser._build_qa_segments(context.document_id, context.documents)
