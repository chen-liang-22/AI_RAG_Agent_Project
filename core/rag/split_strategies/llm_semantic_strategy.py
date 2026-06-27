"""LLM 语义切分策略。

模型只给出原文 start/end 范围和语义标签；
真正写入向量库的正文仍由后端从原文截取，避免模型改写资料内容。
"""

from core.rag.split_strategies.base import BaseSplitStrategy, SplitContext


class LlmSemanticSplitStrategy(BaseSplitStrategy):
    """LLM 语义切片策略。"""

    supported_split_strategies = ("llm_semantic",)

    def split(self, context: SplitContext):
        """由 LLM 给出原文 span，后端按原文截取生成 segment。"""

        return context.parser._build_llm_semantic_segments(
            context.document_id,
            context.documents,
            document_type=context.document_type,
        )
