"""RAG 可观测性日志测试。"""

from langchain_core.documents import Document

from core.rag.query_pipeline import QueryAnalysis
from core.rag.services import rag_service
from core.rag.services.rag_service import RagSummarizeService


class FakeQueryPlanner:
    """测试用查询规划器，固定返回两个检索问题。"""

    def plan_initial(self, query: str, *, trace_id: str | None = None) -> list[str]:
        """返回初始检索问题。"""

        return [query, "扫地机器人保养方式"]


class FakeReranker:
    """测试用精排器，返回前两条资料。"""

    def rerank_by_query(self, **kwargs):
        """模拟精排结果。"""

        grouped_documents = kwargs["grouped_documents"]
        documents = [document for _, group in grouped_documents for document in group]
        return documents[:2]


class FakeRagService(RagSummarizeService):
    """避免连接真实 Qdrant 的 RAG 服务。"""

    def __init__(self):
        super().__init__()
        self.query_planner = FakeQueryPlanner()
        self.reranker = FakeReranker()

    def retrieve_for_queries(
            self,
            queries: list[str],
            analysis: QueryAnalysis,
            *,
            trace_id: str | None = None,
            collection_name: str | None = None,
    ):
        """模拟每个检索问题召回不同数量的资料。"""

        return [
            (
                queries[0],
                [
                    Document(page_content="资料1", metadata={"_vector_score": 0.9, "source_file": "a.txt"}),
                    Document(page_content="资料2", metadata={"_vector_score": 0.8, "source_file": "b.txt"}),
                ],
            ),
            (
                queries[1],
                [Document(page_content="资料3", metadata={"_vector_score": 0.7, "source_file": "c.txt"})],
            ),
        ]


def test_rag_logs_observability_summary(monkeypatch):
    """RAG 检索完成后应输出包含核心观测字段的汇总日志。"""

    log_calls = []
    monkeypatch.setattr(rag_service.logger, "info", lambda *args, **kwargs: log_calls.append(args))
    service = FakeRagService()

    service.retriever_docs("扫地机器人如何保养", trace_id="trace_1", collection_name="agent")

    summary_calls = [args for args in log_calls if args and args[0].startswith("[可观测性] RAG链路汇总")]
    assert summary_calls, "缺少 RAG 可观测性汇总日志"
    template, *values = summary_calls[0]
    assert "追踪编号=%s" in template
    assert "Collection=%s" in template
    assert "检索问题数=%s" in template
    assert "候选资料数=%s" in template
    assert "最终资料数=%s" in template
    assert values[0] == "trace_1"
    assert values[1] == "agent"
    assert values[2] == 2
    assert values[3] == 3
    assert values[4] == 2
