
"""RAG 检索服务。

这个模块是 Agent 工具 `rag_summarize` 背后的核心检索链路。
现在它不再只是“原问题 -> 向量召回 topK”，而是按照设计文档做了一个 MVP 版升级：

1. 默认用原问题直接召回。
2. 根据召回质量判断是否需要 LLM Query Planner 做语义改写/拆分。
3. 多路召回候选资料。
4. 轻量 rerank 精排。
5. 输出更干净的参考资料上下文。

这里仍然不额外调用大模型做中间总结。
这样可以减少流式回答前的等待时间，把模型调用留给最终 Agent 回答。
"""

import json
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document

from rag.knowledge_store import KnowledgeStore
from rag.services.query_planner_service import QueryPlannerService
from rag.query_pipeline import QueryAnalysis
from rag.reranker import RuleBasedReranker
from infrastructure.vector_store_service import VectorStoreService
from utils.config_handler import qdrant_conf, rag_conf
from utils.logger_handler import logger
from utils.qdrant_options import normalize_qdrant_collection_name


@dataclass
class RetrievalQuality:
    """召回质量评估结果。"""

    good_enough: bool
    doc_count: int
    top1_score: float
    top3_avg_score: float
    reason: str


class RagSummarizeService(object):
    """RAG 检索服务类。"""

    def __init__(self):
        self.vector_store: VectorStoreService | None = None  # Qdrant 向量库服务，懒加载，避免 Qdrant 不可用时整个 RAG 初始化失败
        self.vector_stores: dict[str, VectorStoreService] = {}  # 按 collection 缓存向量库服务，避免多知识库互相串用
        self.knowledge_store = KnowledgeStore()  # MySQL 业务元数据仓库，用于文档、会话和系统字典管理
        self.query_planner = QueryPlannerService()  # LLM Query Planner，负责把复杂问题拆成多个 search_query
        self.reranker = RuleBasedReranker()  # 规则版精排器

    def _get_vector_store(self, collection_name: str | None = None) -> VectorStoreService:
        """懒加载 Qdrant 向量库服务。

        Qdrant 是外部服务，可能在开发时暂时没启动。
        如果在 __init__ 里强行初始化，整个 RAG 工具会直接不可用。
        改成懒加载后：
        - Qdrant 可用时，正常做向量召回。
        - Qdrant 不可用时，异常会被检索流程捕获并返回空候选。
        """

        normalized_collection_name = normalize_qdrant_collection_name(collection_name)
        if collection_name is None and self.vector_store is None:
            self.vector_store = VectorStoreService()
            self.vector_stores[normalized_collection_name] = self.vector_store

        if collection_name is None and self.vector_store is not None:
            self.vector_stores[normalized_collection_name] = self.vector_store
            return self.vector_store

        if normalized_collection_name not in self.vector_stores:
            self.vector_stores[normalized_collection_name] = VectorStoreService(collection_name=normalized_collection_name)

        return self.vector_stores[normalized_collection_name]

    def retriever_docs(
            self,
            query: str,
            *,
            history: list[dict] | None = None,
            trace_id: str | None = None,
            collection_name: str | None = None,
    ) -> list[Document]:
        """检索并精排用户问题相关资料。

        这个方法是新版 RAG 的核心：
        - adaptive 模式先用原问题检索。
        - 召回质量差时，才调用 LLM Query Planner 做语义改写/拆分。
        - 多个 search_query 会并行查 Qdrant topK。
        - 最终按用户问题顺序组装上下文，并做去重和数量截断。
        """

        total_start_time = time.perf_counter()
        analysis_start_time = time.perf_counter()
        analysis = self._build_neutral_analysis(query)
        logger.info(
            "[性能] RAG轻量分析完成 追踪编号=%s 耗时毫秒=%.2f 说明=%s",
            trace_id,
            self._elapsed_ms(analysis_start_time),
            "未使用规则意图硬匹配",
        )
        self._log_intent_analysis(query, analysis)
        plan_retrieve_start_time = time.perf_counter()
        search_queries, candidate_groups = self._plan_and_retrieve(
            query,
            analysis,
            history=history,
            trace_id=trace_id,
            collection_name=collection_name,
        )
        logger.info(
            "[性能] RAG召回完成 追踪编号=%s 耗时毫秒=%.2f 检索问题数=%s 候选资料数=%s",
            trace_id,
            self._elapsed_ms(plan_retrieve_start_time),
            len(search_queries),
            sum(len(documents) for _, documents in candidate_groups),
        )
        rerank_start_time = time.perf_counter()
        reranked_docs = self.reranker.rerank_by_query(
            original_query=query,
            grouped_documents=candidate_groups,
            analysis=analysis,
            per_query_keep=int(qdrant_conf.get("per_query_keep", 2) or 2),
            final_context_limit=int(qdrant_conf.get("final_context_limit", 12) or 12),
        )
        logger.info(
            "[性能] RAG精排完成 追踪编号=%s 耗时毫秒=%.2f 最终资料数=%s",
            trace_id,
            self._elapsed_ms(rerank_start_time),
            len(reranked_docs),
        )

        logger.info(
            "[RAG检索] 原问题=%s 意图=%s 候选总数=%s 最终资料数=%s\n检索问题列表：\n%s",
            query,
            analysis.intents,
            sum(len(documents) for _, documents in candidate_groups),
            len(reranked_docs),
            self._format_query_lines(search_queries),
        )
        logger.info("[RAG精排] 最终上下文=%s", self._summarize_documents_for_log(reranked_docs))
        logger.info(
            "[性能] RAG检索精排总耗时 追踪编号=%s 耗时毫秒=%.2f",
            trace_id,
            self._elapsed_ms(total_start_time),
        )

        return reranked_docs

    def _plan_and_retrieve(
            self,
            query: str,
            analysis: QueryAnalysis,
            *,
            history: list[dict] | None = None,
            trace_id: str | None = None,
            collection_name: str | None = None,
    ) -> tuple[list[str], list[tuple[str, list[Document]]]]:
        planner_mode = str(rag_conf.get("query_planner_mode") or "adaptive").strip().lower()

        if planner_mode == "adaptive":
            return self._adaptive_plan_and_retrieve(
                query,
                analysis,
                history=history,
                trace_id=trace_id,
                collection_name=collection_name,
            )

        search_queries = self.query_planner.plan(query, history=history, trace_id=trace_id)
        logger.info(
            "[查询规划] 固定模式查询规划结果 追踪编号=%s 模式=%s 检索问题数=%s\n检索问题列表：\n%s",
            trace_id,
            planner_mode,
            len(search_queries),
            self._format_query_lines(search_queries),
        )
        return search_queries, self.retrieve_for_queries(
            search_queries,
            analysis,
            trace_id=trace_id,
            collection_name=collection_name,
        )

    def _adaptive_plan_and_retrieve(
            self,
            query: str,
            analysis: QueryAnalysis,
            *,
            history: list[dict] | None = None,
            trace_id: str | None = None,
            collection_name: str | None = None,
    ) -> tuple[list[str], list[tuple[str, list[Document]]]]:
        first_queries = self.query_planner.plan_initial(query, trace_id=trace_id)
        if collection_name is None:
            first_groups = self.retrieve_for_queries(first_queries, analysis, trace_id=trace_id)
        else:
            first_groups = self.retrieve_for_queries(
                first_queries,
                analysis,
                trace_id=trace_id,
                collection_name=collection_name,
            )
        first_documents = [document for _, documents in first_groups for document in documents]
        quality = self.evaluate_retrieval_quality(first_documents)
        logger.info(
            "[召回质量] 首次召回评估完成 追踪编号=%s 是否足够=%s 资料数=%s 最高分=%.4f 前三平均分=%.4f 原因=%s",
            trace_id,
            quality.good_enough,
            quality.doc_count,
            quality.top1_score,
            quality.top3_avg_score,
            quality.reason,
        )

        if quality.good_enough:
            logger.info("[查询规划] adaptive模式跳过模型拆分 追踪编号=%s", trace_id)
            return first_queries, first_groups

        if self._should_skip_planner_for_low_quality(analysis, quality):
            logger.info(
                "[查询规划] adaptive模式跳过模型改写 追踪编号=%s 原因=首次召回分数极低 "
                "Collection=%s 最高分=%.4f 前三平均分=%.4f 检索问题数=%s",
                trace_id,
                normalize_qdrant_collection_name(collection_name),
                quality.top1_score,
                quality.top3_avg_score,
                len(first_queries),
            )
            return first_queries, first_groups

        planned_queries = self.query_planner.plan_with_config(query, history=history, trace_id=trace_id)
        search_queries = self.query_planner.merge_queries(first_queries, planned_queries)
        if search_queries == first_queries:
            return first_queries, first_groups

        additional_queries = search_queries[len(first_queries):]
        logger.info(
            "[查询规划] adaptive模式触发模型拆分 追踪编号=%s 总检索问题数=%s 新增检索问题数=%s\n检索问题列表：\n%s",
            trace_id,
            len(search_queries),
            len(additional_queries),
            self._format_query_lines(search_queries),
        )
        logger.info("[查询规划] adaptive模式复用首次原问题召回结果 追踪编号=%s", trace_id)
        if collection_name is None:
            additional_groups = self.retrieve_for_queries(additional_queries, analysis, trace_id=trace_id)
        else:
            additional_groups = self.retrieve_for_queries(
                additional_queries,
                analysis,
                trace_id=trace_id,
                collection_name=collection_name,
            )
        return search_queries, [*first_groups, *additional_groups]

    @staticmethod
    def evaluate_retrieval_quality(documents: list[Document]) -> RetrievalQuality:
        min_docs = int(rag_conf.get("adaptive_retrieve_min_docs", 3) or 3)
        min_score = float(rag_conf.get("adaptive_retrieve_min_score", 0.72) or 0.72)
        min_top3_avg = float(rag_conf.get("adaptive_retrieve_top3_avg_score", 0.68) or 0.68)

        scores = sorted(
            [float(document.metadata.get("_vector_score") or 0.0) for document in documents],
            reverse=True,
        )
        doc_count = len(documents)
        top1_score = scores[0] if scores else 0.0
        top3_scores = scores[:3]
        top3_avg_score = sum(top3_scores) / len(top3_scores) if top3_scores else 0.0

        reasons: list[str] = []
        if doc_count < min_docs:
            reasons.append(f"资料数不足{min_docs}")
        if top1_score < min_score:
            reasons.append(f"最高分低于{min_score}")
        if top3_avg_score < min_top3_avg:
            reasons.append(f"前三平均分低于{min_top3_avg}")

        return RetrievalQuality(
            good_enough=not reasons,
            doc_count=doc_count,
            top1_score=top1_score,
            top3_avg_score=top3_avg_score,
            reason="；".join(reasons) or "召回质量达标",
        )

    def _should_skip_planner_for_low_quality(
            self,
            analysis: QueryAnalysis,
            quality: RetrievalQuality,
    ) -> bool:
        if not bool(rag_conf.get("adaptive_skip_planner_on_very_low_score", True)):
            return False

        if analysis.intents or analysis.filters:
            return False

        max_score = float(rag_conf.get("adaptive_skip_planner_max_score", 0.45) or 0.45)
        max_top3_avg = float(rag_conf.get("adaptive_skip_planner_top3_avg_score", 0.42) or 0.42)
        return quality.top1_score < max_score and quality.top3_avg_score < max_top3_avg

    def debug_retrieve(self, query: str, collection_name: str | None = None) -> dict:
        """返回 RAG 检索调试信息。

        这个方法可以给后续 `/debug/retrieve` 接口使用。
        方便查看：
        - 识别到的意图
        - 生成的子查询
        - metadata filter
        - 精排后的资料来源和分数
        """

        analysis = self._build_neutral_analysis(query)
        self._log_intent_analysis(query, analysis)
        search_queries, candidate_groups = self._plan_and_retrieve(
            query,
            analysis,
            collection_name=collection_name,
        )
        reranked_docs = self.reranker.rerank_by_query(
            original_query=query,
            grouped_documents=candidate_groups,
            analysis=analysis,
            per_query_keep=int(qdrant_conf.get("per_query_keep", 2) or 2),
            final_context_limit=int(qdrant_conf.get("final_context_limit", 12) or 12),
        )

        return {
            "query": query,
            "intents": analysis.intents,
            "sub_queries": search_queries,
            "filters": analysis.filters,
            "candidate_count": sum(len(documents) for _, documents in candidate_groups),
            "groups": [
                {
                    "search_query": search_query,
                    "candidate_count": len(documents),
                }
                for search_query, documents in candidate_groups
            ],
            "reranked": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "vector_score": doc.metadata.get("_vector_score"),
                    "rerank_score": doc.metadata.get("_rerank_score"),
                }
                for doc in reranked_docs
            ],
        }

    def retrieve_for_queries(
            self,
            queries: list[str],
            analysis: QueryAnalysis,
            *,
            trace_id: str | None = None,
            collection_name: str | None = None,
    ) -> list[tuple[str, list[Document]]]:
        """每个 search_query 单独召回 Qdrant top5。"""

        safe_queries = [query for query in queries if query.strip()]
        if not safe_queries:
            return []

        if len(safe_queries) == 1:
            return [
                self._retrieve_one_query(
                    0,
                    safe_queries[0],
                    analysis,
                    trace_id=trace_id,
                    collection_name=collection_name,
                )
            ]

        self._get_vector_store(collection_name)
        max_workers = min(len(safe_queries), int(qdrant_conf.get("parallel_query_workers", 4) or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._retrieve_one_query,
                    query_index,
                    search_query,
                    analysis,
                    trace_id=trace_id,
                    collection_name=collection_name,
                )
                for query_index, search_query in enumerate(safe_queries)
            ]
            return [future.result() for future in futures]

    def _retrieve_one_query(
            self,
            query_index: int,
            search_query: str,
            analysis: QueryAnalysis,
            *,
            trace_id: str | None = None,
            collection_name: str | None = None,
    ) -> tuple[str, list[Document]]:
        per_query_top_k = int(qdrant_conf.get("per_query_top_k", 5) or 5)
        use_metadata_filter = bool(qdrant_conf.get("use_metadata_filter", False))
        documents: list[Document] = []

        if use_metadata_filter and analysis.filters:
            documents = self._search_documents(
                query_index,
                search_query,
                per_query_top_k,
                filters=analysis.filters,
                filter_label="元数据",
                trace_id=trace_id,
                collection_name=collection_name,
            )

        if not documents:
            documents = self._search_documents(
                query_index,
                search_query,
                per_query_top_k,
                filters=None,
                filter_label="无",
                trace_id=trace_id,
                collection_name=collection_name,
            )

        logger.info(
            "[RAG召回] 查询序号=%s 实际召回数=%s 候选资料=%s",
            query_index,
            len(documents),
            self._summarize_documents_for_log(documents),
        )
        self._log_retrieved_scores(search_query, documents)
        return (
            search_query,
            self._annotate_search_query(
                documents,
                search_query=search_query,
                query_index=query_index,
            ),
        )

    def _search_documents(
            self,
            query_index: int,
            search_query: str,
            k: int,
            *,
            filters: dict[str, list[str]] | None,
            filter_label: str,
            trace_id: str | None = None,
            collection_name: str | None = None,
    ) -> list[Document]:
        try:
            retrieve_start_time = time.perf_counter()
            logger.info(
                "[RAG召回] 查询序号=%s 检索问题=%s 过滤条件=%s 召回数量=%s",
                query_index,
                search_query,
                filters or "无",
                k,
            )
            documents = self._get_vector_store(collection_name).search_documents(
                search_query,
                k=k,
                filters=filters,
            )
            logger.info(
                "[性能] 向量召回完成 追踪编号=%s 查询序号=%s 过滤方式=%s 耗时毫秒=%.2f 资料数=%s",
                trace_id,
                query_index,
                filter_label,
                self._elapsed_ms(retrieve_start_time),
                len(documents),
            )
            return documents
        except (ConnectionError, TimeoutError, RuntimeError, ValueError) as exc:
            logger.warning(
                "[RAG召回] 向量检索失败 追踪编号=%s 查询序号=%s 过滤方式=%s 检索问题=%s 错误=%s",
                trace_id,
                query_index,
                filter_label,
                search_query,
                exc,
            )
            return []

    @staticmethod
    def _log_intent_analysis(query: str, analysis: QueryAnalysis) -> None:
        logger.info(
            "[RAG轻量分析] 原问题=%s 意图=%s 关键词=%s 过滤条件=%s 子问题=%s",
            query,
            analysis.intents,
            analysis.keywords,
            analysis.filters,
            analysis.sub_queries,
        )

    @staticmethod
    def _build_neutral_analysis(query: str) -> QueryAnalysis:
        """构造无规则硬匹配的分析结果。

        主链路不再依赖关键词意图、规则子问题或 metadata 强过滤。
        QueryAnalysis 只作为 rerank 和调试接口的兼容数据结构。
        """

        return QueryAnalysis(
            original_query=query.strip(),
            intents=[],
            sub_queries=[query.strip()] if query.strip() else [],
            filters={},
            keywords=[],
        )

    @staticmethod
    def _annotate_search_query(
            documents: list[Document],
            *,
            search_query: str,
            query_index: int,
    ) -> list[Document]:
        annotated_documents: list[Document] = []
        for document in documents:
            metadata = dict(document.metadata)
            metadata["_search_query"] = search_query
            metadata["_search_query_index"] = query_index
            annotated_documents.append(Document(page_content=document.page_content, metadata=metadata))
        return annotated_documents

    @staticmethod
    def _summarize_documents_for_log(documents: list[Document], limit: int = 8) -> list[dict]:
        result: list[dict] = []
        for index, document in enumerate(documents[:limit], start=1):
            metadata = document.metadata
            result.append(
                {
                    "rank": index,
                    "source_file": metadata.get("source_file") or metadata.get("source"),
                    "question_no": metadata.get("question_no"),
                    "content_type": metadata.get("content_type"),
                    "category": metadata.get("category"),
                    "vector_score": metadata.get("_vector_score"),
                    "rerank_score": metadata.get("_rerank_score"),
                    "search_query": metadata.get("_search_query"),
                    "preview": document.page_content.replace("\n", " ")[:120],
                }
            )
        return result

    @staticmethod
    def _log_retrieved_scores(search_query: str, documents: list[Document]) -> None:
        """按检索问题逐行打印召回分值，方便校准 adaptive 阈值。"""

        if not documents:
            logger.info("[RAG召回分值]\n检索问题：%s\n无召回结果", search_query)
            return

        score_lines: list[str] = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            score = float(metadata.get("_vector_score") or 0.0)
            source_file = metadata.get("source_file") or metadata.get("source") or "未知来源"
            question_no = metadata.get("question_no")
            question_text = f" 问题编号={question_no}" if question_no is not None else ""
            score_lines.append(
                f"第{index}条：分值={score:.6f} 来源={source_file}{question_text}"
            )

        logger.info("[RAG召回分值]\n检索问题：%s\n%s", search_query, "\n".join(score_lines))

    @staticmethod
    def _format_query_lines(queries: list[str]) -> str:
        """把检索问题列表格式化为多行日志。"""

        if not queries:
            return "无"
        return "\n".join(f"第{index}个：{query}" for index, query in enumerate(queries, start=1))

    def _keyword_retrieve(self, analysis: QueryAnalysis, limit: int) -> list[Document]:
        """旧版关系型数据库关键词补充召回入口。

        当前回答链路按要求只走 Qdrant。
        这个方法保留为空实现，避免旧代码路径继续使用关系型数据库 LIKE 查询知识正文。
        """

        return []

    @staticmethod
    def _read_metadata_value(metadata_json: str | None, key: str):
        if not metadata_json:
            return None

        try:
            metadata = json.loads(metadata_json)
        except (json.JSONDecodeError, TypeError):
            return None

        return metadata.get(key)

    def rag_summarize(
            self,
            query: str,
            *,
            history: list[dict] | None = None,
            trace_id: str | None = None,
            collection_name: str | None = None,
    ) -> str:
        """返回给 Agent 的 RAG 参考资料文本。

        注意这里不是最终回答。
        它只是把检索到的资料整理成上下文，交给 Agent 再生成自然语言答案。
        """

        start_time = time.perf_counter()
        context_docs = self.retriever_docs(
            query,
            history=history,
            trace_id=trace_id,
            collection_name=collection_name,
        )

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            context += self._format_reference(counter, doc)

        logger.info(
            "[性能] RAG上下文格式化完成 追踪编号=%s 耗时毫秒=%.2f 资料数=%s 上下文字符数=%s",
            trace_id,
            self._elapsed_ms(start_time),
            len(context_docs),
            len(context),
        )
        return context or "未检索到相关参考资料。"

    @staticmethod
    def _format_reference(index: int, doc: Document) -> str:
        """把单条 Document 格式化成干净的参考资料。"""

        metadata = doc.metadata
        source_file = metadata.get("source_file") or metadata.get("source") or "未知来源"
        source_page = metadata.get("source_page") or metadata.get("page")
        unit_type = metadata.get("unit_type") or "text"
        category = metadata.get("category") or "通用知识"

        source_parts = [
            f"来源文件：{source_file}",
            f"知识类型：{unit_type}",
            f"分类：{category}",
        ]

        if source_page is not None:
            source_parts.append(f"页码：{source_page}")

        return (
            f"【参考资料{index}】\n"
            f"{doc.page_content}\n"
            f"来源：{'；'.join(source_parts)}\n\n"
        )

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000
