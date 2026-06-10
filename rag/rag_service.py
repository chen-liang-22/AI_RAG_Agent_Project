
"""RAG 检索服务。

这个模块是 Agent 工具 `rag_summarize` 背后的核心检索链路。
现在它不再只是“原问题 -> 向量召回 topK”，而是按照设计文档做了一个 MVP 版升级：

1. 规则版多意图识别。
2. 根据意图生成子查询。
3. 根据意图生成 Qdrant metadata filter。
4. 多路召回候选资料。
5. 规则版 rerank 精排。
6. 输出更干净的参考资料上下文。

这里仍然不额外调用大模型做中间总结。
这样可以减少流式回答前的等待时间，把模型调用留给最终 Agent 回答。
"""

import re

from langchain_core.documents import Document

from rag.query_pipeline import QueryAnalysis, RuleBasedIntentAnalyzer
from rag.reranker import RuleBasedReranker
from rag.vector_store import VectorStoreService
from utils.config_handler import qdrant_conf
from utils.logger_handler import logger


class RagSummarizeService(object):
    """RAG 检索服务类。"""

    def __init__(self):
        self.vector_store: VectorStoreService | None = None  # Qdrant 向量库服务，懒加载，避免 Qdrant 不可用时整个 RAG 初始化失败
        self.intent_analyzer = RuleBasedIntentAnalyzer()  # 规则版多意图分析器
        self.reranker = RuleBasedReranker()  # 规则版精排器

    def try_answer_exact_faq_query(self, query: str) -> str | None:
        """尝试直接回答 FAQ 编号/清单类问题。

        这类问题不适合优先走向量相似度搜索，例如：
        - “扫拖一体机器人100问第95问是什么？”
        - “扫拖一体机器人100问都有哪些？”

        FAQ 文档入库后已经抽取到 SQLite 的 faq_items 表，因此这里直接做
        结构化查询，能避免向量召回只召到片段导致“明明有第95问却答不上来”。
        返回 None 表示当前问题不是 FAQ 精确查询，继续走普通 Agent/RAG。
        """

        normalized_query = query.strip()
        if not normalized_query:
            return None

        if self._is_faq_list_query(normalized_query):
            return self.list_faq_questions(document_hint=normalized_query)

        question_no = self._extract_faq_question_no(normalized_query)
        if question_no is not None:
            return self.get_faq_item_by_number(question_no=question_no, document_hint=normalized_query)

        return self.get_faq_item_by_question(normalized_query)

    def get_faq_item_by_number(self, question_no: int, document_hint: str | None = None) -> str:
        """按编号从 Qdrant payload 中查询 FAQ/100问 的某一问。"""

        documents = VectorStoreService.scroll_faq_documents(
            question_no=question_no,
            limit=100,
        )
        document = self._select_best_faq_document(documents, document_hint)
        if document is None:
            return (
                f"没有在 Qdrant FAQ 向量中找到第{question_no}问。\n"
                "如果原文件确实包含这一问，请重新执行 /knowledge/reload 重建向量索引。"
            )

        return self._format_faq_document(document)

    def list_faq_questions(self, document_hint: str | None = None, limit: int = 120) -> str:
        """从 Qdrant payload 中列出 FAQ/100问 文档的问题清单。"""

        documents = VectorStoreService.scroll_faq_documents(limit=2000)
        source_file = self._select_best_faq_source_file(documents, document_hint)
        if source_file:
            documents = [doc for doc in documents if doc.metadata.get("source_file") == source_file]

        if not documents:
            return (
                "没有在 Qdrant FAQ 向量中找到可列出的问题清单。\n"
                "如果原文件已经上传过，请重新执行 /knowledge/reload 重建向量索引。"
            )

        final_limit = max(1, min(int(limit), 300))
        documents.sort(key=lambda doc: int(doc.metadata.get("question_no") or 999999))
        selected_documents = documents[:final_limit]

        lines = [
            f"来源文件：{source_file or selected_documents[0].metadata.get('source_file') or '未知来源'}",
            f"共{len(documents)}问，当前列出{len(selected_documents)}问：",
        ]
        for document in selected_documents:
            question_no = document.metadata.get("question_no")
            question = document.metadata.get("question") or self._extract_question_from_faq_content(document.page_content)
            prefix = f"{question_no}. " if question_no is not None else "- "
            lines.append(f"{prefix}{question}")

        if len(documents) > len(selected_documents):
            lines.append(f"还有{len(documents) - len(selected_documents)}问未展示，可提高 limit 参数继续查询。")

        return "\n".join(lines)

    def get_faq_item_by_question(self, query: str) -> str | None:
        """按语义从 Qdrant FAQ 向量中查询答案。

        用户不会总是复述知识库原句，所以这里不做 LIKE，也不做同义词表。
        直接在 Qdrant 的 FAQ 向量子集里搜索，命中高置信结果后返回 payload。
        """

        if len(query.strip()) < 4:
            return None

        documents = self._get_vector_store().search_faq_documents(query, k=8)
        if not documents:
            return None

        best_document = documents[0]
        vector_score = float(best_document.metadata.get("_vector_score") or 0.0)
        if vector_score < 0.62:
            return None

        return self._format_faq_document(best_document)

    def _get_vector_store(self) -> VectorStoreService:
        """懒加载 Qdrant 向量库服务。

        Qdrant 是外部服务，可能在开发时暂时没启动。
        如果在 __init__ 里强行初始化，整个 RAG 工具会直接不可用。
        改成懒加载后：
        - Qdrant 可用时，正常做向量召回。
        - Qdrant 不可用时，异常会被 _multi_retrieve 捕获，继续走 SQLite 关键词召回。
        """

        if self.vector_store is None:
            self.vector_store = VectorStoreService()

        return self.vector_store

    def retriever_docs(self, query: str) -> list[Document]:
        """检索并精排用户问题相关资料。

        这个方法是新版 RAG 的核心：
        - 先分析用户问题属于选购、故障、维护等哪类意图。
        - 再用原问题和子查询做多路召回。
        - 有意图过滤条件时，优先查更匹配的 unit_type/category。
        - 额外补一次无过滤召回，避免规则过滤过窄导致漏召回。
        - 最后用规则 rerank 排序并截断到 qdrant_conf["k"]。
        """

        analysis = self.intent_analyzer.analyze(query)
        candidate_docs = self._multi_retrieve(analysis)
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=candidate_docs,
            analysis=analysis,
            limit=qdrant_conf["k"],
        )

        logger.info(
            "[rag] query=%s intents=%s sub_queries=%s candidates=%s final=%s",
            query,
            analysis.intents,
            analysis.sub_queries,
            len(candidate_docs),
            len(reranked_docs),
        )

        return reranked_docs

    def debug_retrieve(self, query: str) -> dict:
        """返回 RAG 检索调试信息。

        这个方法可以给后续 `/debug/retrieve` 接口使用。
        方便查看：
        - 识别到的意图
        - 生成的子查询
        - metadata filter
        - 精排后的资料来源和分数
        """

        analysis = self.intent_analyzer.analyze(query)
        candidate_docs = self._multi_retrieve(analysis)
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=candidate_docs,
            analysis=analysis,
            limit=qdrant_conf["k"],
        )

        return {
            "query": query,
            "intents": analysis.intents,
            "sub_queries": analysis.sub_queries,
            "filters": analysis.filters,
            "candidate_count": len(candidate_docs),
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

    def _multi_retrieve(self, analysis: QueryAnalysis) -> list[Document]:
        """根据 QueryAnalysis 做多路召回。"""

        candidate_docs: list[Document] = []
        candidate_k = max(qdrant_conf["k"] * 4, 8)

        for sub_query in analysis.sub_queries:
            if analysis.filters:
                # 第一轮：带 metadata filter 的定向召回。
                # 例如用户问“迷路怎么办”，优先查 troubleshooting/faq。
                try:
                    candidate_docs.extend(
                        self._get_vector_store().search_documents(
                            sub_query,
                            k=candidate_k,
                            filters=analysis.filters,
                        )
                    )
                except Exception as exc:
                    logger.warning(f"[rag] vector retrieve with filters failed: {exc}")

            # 第二轮：无过滤召回。
            # 这是兜底召回，避免规则意图判断不准时漏掉相关资料。
            try:
                candidate_docs.extend(
                    self._get_vector_store().search_documents(
                        sub_query,
                        k=candidate_k,
                        filters=None,
                    )
                )
            except Exception as exc:
                logger.warning(f"[rag] vector retrieve without filters failed: {exc}")

        return candidate_docs

    def _keyword_retrieve(self, analysis: QueryAnalysis, limit: int) -> list[Document]:
        """从 SQLite 做关键词补充召回。

        当前回答链路按要求只走 Qdrant。
        这个方法保留为空实现，避免旧代码路径继续使用 SQLite LIKE 查询。
        """

        return []

    @staticmethod
    def _read_metadata_value(metadata_json: str | None, key: str):
        if not metadata_json:
            return None

        try:
            import json

            metadata = json.loads(metadata_json)
        except Exception:
            return None

        return metadata.get(key)

    @staticmethod
    def _extract_faq_question_no(query: str) -> int | None:
        """从“第95问是什么/95问是什么”这类表达中抽取编号。"""

        explicit_matches = list(re.finditer(r"第\s*(\d{1,4})\s*[问題题]", query))
        if explicit_matches:
            return int(explicit_matches[-1].group(1))

        implicit_matches = list(
            re.finditer(r"(?<!\d)(\d{1,4})\s*[问題题](?![都全一共总有清列所])", query)
        )
        if len(implicit_matches) >= 2:
            return int(implicit_matches[-1].group(1))

        if len(implicit_matches) == 1:
            number = int(implicit_matches[0].group(1))
            if number == 100 and "第" not in query:
                return None
            return number

        return None

    @staticmethod
    def _is_faq_list_query(query: str) -> bool:
        """判断是否在询问 FAQ/100问 的问题清单。"""

        mentions_faq_document = any(word in query for word in ["100问", "问答", "FAQ", "faq", "常见问题"])
        asks_for_list = any(
            word in query
            for word in ["都有哪些", "有哪些", "有哪", "哪100问", "哪一百问", "全部", "清单", "列表", "所有"]
        )
        return mentions_faq_document and asks_for_list

    @staticmethod
    def _normalize_question_text(value: str) -> str:
        ignore_chars = set(" \t\r\n，。！？?：:；;、,.()（）【】[]“”\"'")
        return "".join(char.lower() for char in value.strip() if char not in ignore_chars)

    @classmethod
    def _select_best_faq_document(
            cls,
            documents: list[Document],
            document_hint: str | None,
    ) -> Document | None:
        if not documents:
            return None

        clean_hint = (document_hint or "").strip()
        if not clean_hint:
            return documents[0]

        return max(
            documents,
            key=lambda doc: cls._document_hint_score(clean_hint, doc.metadata.get("source_file") or ""),
        )

    @classmethod
    def _select_best_faq_source_file(
            cls,
            documents: list[Document],
            document_hint: str | None,
    ) -> str | None:
        source_files = sorted({doc.metadata.get("source_file") for doc in documents if doc.metadata.get("source_file")})
        if not source_files:
            return None

        clean_hint = (document_hint or "").strip()
        if not clean_hint:
            return source_files[0]

        return max(source_files, key=lambda source_file: cls._document_hint_score(clean_hint, source_file))

    @staticmethod
    def _document_hint_score(document_hint: str, source_file: str) -> float:
        hint = RagSummarizeService._normalize_question_text(document_hint)
        name = RagSummarizeService._normalize_question_text(source_file.rsplit(".", 1)[0])

        if not hint or not name:
            return 0.0

        if name in hint or hint in name:
            return 100.0

        ignore_chars = set("的是了和与及第问题什么哪些全部都有一下一个这个那个")
        hint_chars = {char for char in hint if char not in ignore_chars}
        name_chars = {char for char in name if char not in ignore_chars}
        if not hint_chars or not name_chars:
            return 0.0

        return len(hint_chars & name_chars) / max(len(name_chars), 1)

    @staticmethod
    def _format_faq_document(document: Document) -> str:
        metadata = document.metadata
        source_file = metadata.get("source_file") or metadata.get("source") or "未知来源"
        question_no = metadata.get("question_no")
        question = metadata.get("question") or RagSummarizeService._extract_question_from_faq_content(document.page_content)
        answer = RagSummarizeService._extract_answer_from_faq_content(document.page_content)
        category = metadata.get("category") or metadata.get("heading_path")

        question_prefix = f"第{question_no}问：" if question_no is not None else "问题："
        answer_text = f"\n答案：{answer}" if answer else ""
        category_text = f"\n分类：{category}" if category else ""
        return (
            f"来源文件：{source_file}\n"
            f"{question_prefix}{question}"
            f"{answer_text}"
            f"{category_text}"
        )

    @staticmethod
    def _extract_question_from_faq_content(content: str) -> str:
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("问题："):
                return line.removeprefix("问题：").strip()
        return content.strip().splitlines()[0] if content.strip() else ""

    @staticmethod
    def _extract_answer_from_faq_content(content: str) -> str:
        answer_lines = []
        collecting = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("答案："):
                collecting = True
                answer_lines.append(stripped.removeprefix("答案：").strip())
                continue
            if collecting:
                answer_lines.append(stripped)
        return "\n".join(line for line in answer_lines if line).strip()

    def rag_summarize(self, query: str) -> str:
        """返回给 Agent 的 RAG 参考资料文本。

        注意这里不是最终回答。
        它只是把检索到的资料整理成上下文，交给 Agent 再生成自然语言答案。
        """

        context_docs = self.retriever_docs(query)

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            context += self._format_reference(counter, doc)

        return context or "未检索到相关参考资料。"

    @staticmethod
    def _format_reference(index: int, doc: Document) -> str:
        """把单条 Document 格式化成干净的参考资料。"""

        metadata = doc.metadata
        source_file = metadata.get("source_file") or metadata.get("source") or "未知来源"
        source_page = metadata.get("source_page") or metadata.get("page")
        unit_type = metadata.get("unit_type") or "general"
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


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
