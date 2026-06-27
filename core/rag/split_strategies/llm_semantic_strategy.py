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
