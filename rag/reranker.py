from langchain_core.documents import Document

from rag.query_pipeline import QueryAnalysis


class RuleBasedReranker:
    """轻量规则精排器。

    设计文档里提到 rerank 的核心目标是：
    “向量召回先找候选，精排再决定哪些内容最适合放进最终上下文。”

    第一版不接额外模型，避免增加部署成本和响应延迟。
    当前打分由几部分组成：
    - Qdrant 向量相似度分数。
    - 用户问题关键词是否命中文本内容。
    - 关键词是否命中文件名、分类、知识单元类型。
    - 检索意图是否和知识单元类型匹配。
    """

    # 意图到知识单元类型的匹配关系。
    # 如果命中，说明这条资料更符合用户问题的业务方向。
    intent_unit_type_bonus: dict[str, list[str]] = {
        "purchase": ["guide", "faq"],
        "comparison": ["guide", "faq"],
        "troubleshooting": ["troubleshooting", "faq"],
        "maintenance": ["maintenance", "faq"],
    }

    intent_negative_keywords: dict[str, list[str]] = {
        "troubleshooting": ["导航技术", "选购", "参数", "品牌对比"],
        "purchase": ["故障现象", "修复", "检测："],
        "maintenance": ["选购", "品牌对比"],
    }

    def rerank(self, query: str, documents: list[Document], analysis: QueryAnalysis, limit: int) -> list[Document]:
        """对候选文档去重、打分、排序，并返回前 limit 条。"""

        unique_documents = self._deduplicate_documents(documents)
        scored_documents = [
            (self._score_document(query, document, analysis), document)
            for document in unique_documents
        ]
        scored_documents.sort(key=lambda item: item[0], reverse=True)

        reranked_documents: list[Document] = []
        for score, document in scored_documents[:limit]:
            metadata = dict(document.metadata)
            metadata["_rerank_score"] = round(score, 4)
            reranked_documents.append(Document(page_content=document.page_content, metadata=metadata))

        return reranked_documents

    def _score_document(self, query: str, document: Document, analysis: QueryAnalysis) -> float:
        """计算单条候选资料的规则分数。"""

        metadata = document.metadata
        content = document.page_content
        title = str(metadata.get("title") or metadata.get("source_file") or "")
        category = str(metadata.get("category") or "")
        unit_type = str(metadata.get("unit_type") or "")
        vector_score = float(metadata.get("_vector_score") or 0.0)

        score = vector_score

        if metadata.get("_keyword_hit"):
            score += 0.45

        for keyword in analysis.keywords:
            if not keyword:
                continue

            if keyword in content:
                score += 0.18
            if keyword in title:
                score += 0.25
            if keyword in category:
                score += 0.22
            if keyword in unit_type:
                score += 0.15

        for intent in analysis.intents:
            if unit_type in self.intent_unit_type_bonus.get(intent, []):
                score += 0.35
            for negative_keyword in self.intent_negative_keywords.get(intent, []):
                if negative_keyword in content:
                    score -= 0.25

        # 如果原问题中的完整短句能在内容里出现，说明这条资料高度相关。
        if query and query in content:
            score += 0.4

        return score

    @staticmethod
    def _deduplicate_documents(documents: list[Document]) -> list[Document]:
        """按 unit_id/chunk_id/source+content 去重。

        多路召回会对原问题和多个子查询分别检索。
        同一个 Qdrant point 很可能被多次召回，所以进入精排前先去重。
        """

        result: list[Document] = []
        seen: set[str] = set()

        for document in documents:
            metadata = document.metadata
            unit_id = metadata.get("unit_id")
            chunk_id = metadata.get("chunk_id")

            if unit_id:
                unique_key = str(unit_id)
            elif chunk_id:
                unique_key = str(chunk_id)
            else:
                unique_key = f"{metadata.get('source')}::{metadata.get('page')}::{document.page_content[:80]}"

            if unique_key in seen:
                continue

            seen.add(unique_key)
            result.append(document)

        return result
