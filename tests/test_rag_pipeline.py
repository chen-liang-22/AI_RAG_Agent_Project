from langchain_core.documents import Document

from rag.query_pipeline import QueryAnalysis
from rag.rag_service import RagSummarizeService
from rag.reranker import RuleBasedReranker
from rag.vector_store import VectorStoreService


def test_build_metadata_filter_uses_qdrant_metadata_prefix():
    qdrant_filter = VectorStoreService.build_metadata_filter(
        {
            "unit_type": ["qa", ""],
            "category": ["常见问答", "选购指南"],
        }
    )

    assert qdrant_filter is not None
    assert [condition.key for condition in qdrant_filter.must] == [
        "metadata.unit_type",
        "metadata.category",
    ]
    assert qdrant_filter.must[0].match.any == ["qa"]


def test_rerank_by_query_keeps_query_coverage_and_deduplicates():
    reranker = RuleBasedReranker()
    analysis = QueryAnalysis(
        original_query="扫地机器人选购和故障",
        intents=["purchase", "problem"],
        keywords=["选购", "故障"],
    )
    shared = Document(page_content="扫地机器人选购建议", metadata={"unit_id": "u1", "_vector_score": 0.7})
    grouped_documents = [
        (
            "选购",
            [
                shared,
                Document(page_content="吸力和导航选购", metadata={"unit_id": "u2", "_vector_score": 0.6}),
            ],
        ),
        (
            "故障",
            [
                shared,
                Document(page_content="故障排查和修复", metadata={"unit_id": "u3", "_vector_score": 0.8}),
            ],
        ),
    ]

    result = reranker.rerank_by_query(
        original_query="扫地机器人选购和故障",
        grouped_documents=grouped_documents,
        analysis=analysis,
        per_query_keep=1,
        final_context_limit=4,
    )

    unit_ids = [document.metadata["unit_id"] for document in result]
    assert len(unit_ids) == len(set(unit_ids))
    assert len(result) >= 2
    assert {document.metadata["_search_query"] for document in result[:2]} == {"选购", "故障"}


def test_retrieval_quality_accepts_stable_high_score_results():
    documents = [
        Document(page_content="大户型选购", metadata={"_vector_score": 0.80}),
        Document(page_content="续航", metadata={"_vector_score": 0.74}),
        Document(page_content="自动集尘", metadata={"_vector_score": 0.72}),
    ]

    quality = RagSummarizeService.evaluate_retrieval_quality(documents)

    assert quality.good_enough is True
    assert quality.doc_count == 3


def test_retrieval_quality_rejects_low_score_results():
    documents = [
        Document(page_content="不相关内容", metadata={"_vector_score": 0.60}),
        Document(page_content="另一个内容", metadata={"_vector_score": 0.58}),
    ]

    quality = RagSummarizeService.evaluate_retrieval_quality(documents)

    assert quality.good_enough is False
    assert "资料数不足" in quality.reason


def test_adaptive_plan_skips_llm_when_initial_retrieval_is_good(monkeypatch):
    service = RagSummarizeService()
    analysis = QueryAnalysis(original_query="大户型适合哪些扫地机器人", intents=["purchase"])
    documents = [
        Document(page_content="大户型选购", metadata={"_vector_score": 0.80}),
        Document(page_content="续航", metadata={"_vector_score": 0.74}),
        Document(page_content="自动集尘", metadata={"_vector_score": 0.72}),
    ]

    monkeypatch.setattr(service, "retrieve_for_queries", lambda queries, analysis, trace_id=None: [(queries[0], documents)])
    monkeypatch.setattr(
        service.query_planner,
        "plan_with_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应调用模型查询规划")),
    )

    queries, groups = service._adaptive_plan_and_retrieve("大户型适合哪些扫地机器人", analysis)

    assert queries == ["大户型适合哪些扫地机器人"]
    assert groups == [("大户型适合哪些扫地机器人", documents)]


def test_adaptive_plan_calls_llm_when_initial_retrieval_is_weak(monkeypatch):
    service = RagSummarizeService()
    analysis = QueryAnalysis(original_query="复杂问题", intents=["purchase"])
    weak_documents = [Document(page_content="弱结果", metadata={"_vector_score": 0.40})]
    strong_documents = [
        Document(page_content="补救结果", metadata={"_vector_score": 0.82}),
        Document(page_content="更多结果", metadata={"_vector_score": 0.75}),
        Document(page_content="覆盖结果", metadata={"_vector_score": 0.72}),
    ]
    calls: list[list[str]] = []

    def fake_retrieve(queries, analysis, trace_id=None):
        calls.append(list(queries))
        if len(calls) == 1:
            return [(queries[0], weak_documents)]
        return [(query, strong_documents) for query in queries]

    monkeypatch.setattr(service, "retrieve_for_queries", fake_retrieve)
    monkeypatch.setattr(service.query_planner, "plan_with_config", lambda *args, **kwargs: ["宠物家庭选购", "地毯清扫"])

    queries, groups = service._adaptive_plan_and_retrieve("复杂问题", analysis)

    assert queries == ["复杂问题", "宠物家庭选购", "地毯清扫"]
    assert len(calls) == 2
    assert len(groups) == 3


def test_adaptive_plan_skips_llm_when_score_is_very_low_without_business_keyword(monkeypatch):
    service = RagSummarizeService()
    analysis = QueryAnalysis(original_query="reacjhnio")
    weak_documents = [
        Document(page_content="unrelated one", metadata={"_vector_score": 0.38}),
        Document(page_content="unrelated two", metadata={"_vector_score": 0.37}),
        Document(page_content="unrelated three", metadata={"_vector_score": 0.36}),
    ]

    monkeypatch.setattr(service, "retrieve_for_queries", lambda queries, analysis, trace_id=None: [(queries[0], weak_documents)])
    monkeypatch.setattr(
        service.query_planner,
        "plan_with_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("planner should be skipped")),
    )

    queries, groups = service._adaptive_plan_and_retrieve("reacjhnio", analysis)

    assert queries == ["reacjhnio"]
    assert groups == [("reacjhnio", weak_documents)]


def test_adaptive_plan_keeps_llm_for_business_keyword_even_when_score_is_low(monkeypatch):
    service = RagSummarizeService()
    analysis = QueryAnalysis(original_query="没电还能跑吗")
    weak_documents = [
        Document(page_content="weak one", metadata={"_vector_score": 0.38}),
        Document(page_content="weak two", metadata={"_vector_score": 0.37}),
        Document(page_content="weak three", metadata={"_vector_score": 0.36}),
    ]
    calls: list[list[str]] = []

    def fake_retrieve(queries, analysis, trace_id=None):
        calls.append(list(queries))
        return [(query, weak_documents) for query in queries]

    monkeypatch.setattr(service, "retrieve_for_queries", fake_retrieve)
    monkeypatch.setattr(service.query_planner, "plan_with_config", lambda *args, **kwargs: ["低电量清扫能力"])

    queries, groups = service._adaptive_plan_and_retrieve("没电还能跑吗", analysis)

    assert queries == ["没电还能跑吗", "低电量清扫能力"]
    assert len(calls) == 2
    assert len(groups) == 2


def test_retrieve_one_query_skips_metadata_filter_when_disabled(monkeypatch):
    service = RagSummarizeService()
    calls = []

    class FakeVectorStore:
        def search_documents(self, query, *, k=None, filters=None):
            calls.append(filters)
            return [Document(page_content="结果", metadata={"_vector_score": 0.8})]

    service.vector_store = FakeVectorStore()
    analysis = QueryAnalysis(
        original_query="大户型",
        filters={"unit_type": ["numbered"]},
    )

    _, documents = service._retrieve_one_query(0, "大户型", analysis)

    assert calls == [None]
    assert len(documents) == 1
