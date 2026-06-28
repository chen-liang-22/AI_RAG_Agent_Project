"""销售训练检索服务测试。"""

from langchain_core.documents import Document

from app.application.training.training_query_service import TrainingQueryService


class FakeTrainingRepository:
    """测试用训练仓储，只返回当前发布批次。"""

    def __init__(self, batch_ids: list[str]):
        self.batch_ids = batch_ids

    def list_current_published_batch_ids(self) -> list[str]:
        """返回测试批次编号。"""

        return self.batch_ids


class FakeVectorService:
    """测试用向量服务，记录调用参数并返回固定文档。"""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.last_query = ""
        self.last_k = 0
        self.last_filters = {}

    def search_documents(self, query: str, *, k: int, filters: dict) -> list[Document]:
        """模拟 Qdrant 检索。"""

        self.last_query = query
        self.last_k = k
        self.last_filters = filters
        return self.documents


def test_search_training_evidence_filters_current_batches_and_visibility():
    """只检索当前发布批次，并按 visibility 过滤证据。"""

    documents = [
        Document(
            page_content="A" * 900,
            metadata={
                "chunk_id": "chunk_visible",
                "case_part": "客户异议",
                "visibility": "visible",
                "source_file": "case.docx",
                "_vector_score": 0.91,
            },
        ),
        Document(
            page_content="隐藏内容",
            metadata={
                "chunk_id": "chunk_hidden",
                "visibility": "hidden",
                "source_file": "case.docx",
            },
        ),
    ]
    repository = FakeTrainingRepository(["batch_1", "batch_2"])
    vector_service = FakeVectorService(documents)
    service = TrainingQueryService(
        repository=repository,
        vector_service=vector_service,
        collection_name="sales_training_cases",
    )

    evidence = service.search_training_evidence("客户说太贵", visibility=("visible",), k=3)

    assert vector_service.last_query == "客户说太贵"
    assert vector_service.last_k == 3
    assert vector_service.last_filters == {"batch_id": ["batch_1", "batch_2"]}
    assert len(evidence) == 1
    assert evidence[0]["chunk_id"] == "chunk_visible"
    assert evidence[0]["case_part"] == "客户异议"
    assert evidence[0]["visibility"] == "visible"
    assert evidence[0]["score"] == 0.91
    assert evidence[0]["source_file"] == "case.docx"
    assert len(evidence[0]["content"]) == 800


def test_search_training_evidence_skips_vector_query_without_current_batches():
    """没有当前发布批次时不访问向量库。"""

    repository = FakeTrainingRepository([])
    vector_service = FakeVectorService([])
    service = TrainingQueryService(
        repository=repository,
        vector_service=vector_service,
        collection_name="sales_training_cases",
    )

    evidence = service.search_training_evidence("任意问题", visibility=("visible",), k=3)

    assert evidence == []
    assert vector_service.last_query == ""
