from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.document_parser import DocumentParser


def test_faq_parser_splits_numbered_questions_and_answers():
    parser = DocumentParser(
        RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", "。", " ", ""],
        )
    )
    documents = [
        Document(
            page_content=(
                "### 拖扫功能融合类\n"
                "1. 扫拖一体机器人适合小户型吗？\n"
                "- 答：适合，重点看避障和水箱容量。\n"
                "2. 地毯可以拖地吗？\n"
                "- 答：需要支持地毯识别和拖布抬升。"
            ),
            metadata={"page": 1, "source": "faq.txt"},
        )
    ]

    segments, faqs = parser.build_segments_and_faqs(
        document_id="doc_test",
        documents=documents,
        document_type="faq",
        split_strategy="numbered_qa",
    )

    assert len(segments) == 2
    assert len(faqs) == 2
    assert faqs[0].question == "扫拖一体机器人适合小户型吗？"
    assert faqs[0].answer == "适合，重点看避障和水箱容量。"
    assert faqs[0].category == "拖扫功能融合类"
    assert segments[0].metadata["split_strategy"] == "numbered_qa"


def test_document_type_detection_identifies_troubleshooting_text():
    parser = DocumentParser(
        RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", "。", " ", ""],
        )
    )

    detection = parser.detect_document_type(
        "故障排除.txt",
        "1. 故障现象：机器人迷路。检测：检查传感器。修复：清理传感器。",
    )

    assert detection.document_type == "troubleshooting"
    assert detection.split_strategy == "numbered_segments"
