from langchain_core.documents import Document

from core.rag.query_pipeline import QueryAnalysis


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
        "purchase": ["qa", "numbered"],
        "comparison": ["qa", "numbered"],
        "problem": ["qa", "numbered"],
        "usage": ["qa", "numbered"],
    }

    intent_negative_keywords: dict[str, list[str]] = {
        "problem": ["导航技术", "选购", "参数", "品牌对比"],
        "purchase": ["故障现象", "修复", "检测："],
        "usage": ["选购", "品牌对比"],
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

    def rerank_by_query(
            self,
            *,
            original_query: str,
            grouped_documents: list[tuple[str, list[Document]]],
            analysis: QueryAnalysis,
            per_query_keep: int = 2,
            final_context_limit: int = 12,
    ) -> list[Document]:
        """按 search_query 分组精排，并保证每个子问题优先有资料覆盖。

        设计目标：
        - 每个 search_query 先独立精排。
        - 每个 search_query 保留 top2。
        - 最终按用户问题拆分顺序组织上下文。
        - 对重复 Qdrant point 去重。
        """

        selected_documents: list[Document] = []
        seen: set[str] = set()
        safe_per_query_keep = max(1, per_query_keep)
        safe_final_limit = max(1, final_context_limit)

        for query_index, (search_query, documents) in enumerate(grouped_documents):
            unique_documents = self._deduplicate_documents(documents)
            scored_documents = [
                (self._score_document(search_query, document, analysis), document)
                for document in unique_documents
            ]
            scored_documents.sort(key=lambda item: item[0], reverse=True)

            kept_for_query = 0
            for score, document in scored_documents:
                unique_key = self._document_key(document)
                if unique_key in seen:
                    continue

                metadata = dict(document.metadata)
                metadata["_rerank_score"] = round(score, 4)
                metadata["_search_query"] = search_query
                metadata["_search_query_index"] = query_index
                selected_documents.append(Document(page_content=document.page_content, metadata=metadata))
                seen.add(unique_key)
                kept_for_query += 1

                if kept_for_query >= safe_per_query_keep:
                    break
                if len(selected_documents) >= safe_final_limit:
                    return selected_documents

        if len(selected_documents) >= safe_final_limit:
            return selected_documents[:safe_final_limit]

        # 如果部分 query 因重复去重导致资料不足，用全局高分候选补齐。
        overflow_candidates: list[tuple[float, Document]] = []
        for search_query, documents in grouped_documents:
            for document in self._deduplicate_documents(documents):
                unique_key = self._document_key(document)
                if unique_key in seen:
                    continue
                overflow_candidates.append((self._score_document(original_query or search_query, document, analysis), document))

        overflow_candidates.sort(key=lambda item: item[0], reverse=True)
        for score, document in overflow_candidates:
            unique_key = self._document_key(document)
            if unique_key in seen:
                continue

            metadata = dict(document.metadata)
            metadata["_rerank_score"] = round(score, 4)
            metadata.setdefault("_search_query", original_query)
            selected_documents.append(Document(page_content=document.page_content, metadata=metadata))
            seen.add(unique_key)

            if len(selected_documents) >= safe_final_limit:
                break

        return selected_documents

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
            unique_key = RuleBasedReranker._document_key(document)

            if unique_key in seen:
                continue

            seen.add(unique_key)
            result.append(document)

        return result

    @staticmethod
    def _document_key(document: Document) -> str:
        metadata = document.metadata
        point_id = metadata.get("_point_id")
        unit_id = metadata.get("unit_id")
        chunk_id = metadata.get("chunk_id")
        segment_id = metadata.get("segment_id")

        if point_id:
            return f"point:{point_id}"
        if unit_id:
            return f"unit:{unit_id}"
        if chunk_id:
            return f"chunk:{chunk_id}"
        if segment_id:
            return f"segment:{segment_id}"
        return f"{metadata.get('source_file') or metadata.get('source')}::{metadata.get('page')}::{document.page_content[:80]}"
