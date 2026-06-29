from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytest

from app.application import exam_service
from app.infrastructure.vector_store_service import VectorStoreService
from core.rag.document_parser import DocumentParser


def _parser() -> DocumentParser:
    """创建测试用文档解析器。"""

    return DocumentParser(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0))


def test_upload_recommendation_should_prefer_outline_qa_for_pdf_outline_questions():
    """PDF 目录满足章节到问题结构时，上传推荐应优先使用目录问答切分。"""

    outline = [
        {"level": 0, "title": "一、Java 基础", "page": 1},
        {"level": 1, "title": "1、什么是 JVM？", "page": 2},
        {"level": 1, "title": "2、什么是 JDK？", "page": 3},
        {"level": 1, "title": "3、什么是 JRE？", "page": 4},
        {"level": 0, "title": "二、集合", "page": 5},
        {"level": 1, "title": "1、ArrayList 和 LinkedList 有什么区别？", "page": 6},
        {"level": 1, "title": "2、HashMap 的底层结构是什么？", "page": 7},
        {"level": 1, "title": "3、ConcurrentHashMap 如何保证线程安全？", "page": 8},
    ]
    sample_text = "\n".join(
        [
            "1、什么是 JVM？",
            "JVM 是 Java 虚拟机。",
            "2、什么是 JDK？",
            "JDK 是 Java 开发工具包。",
        ]
    )

    detection = _parser().detect_document_type("Java面试题.pdf", sample_text, outline=outline)

    assert detection.document_type == "qa"
    assert detection.split_strategy == "outline_qa"


def test_outline_qa_metadata_should_keep_first_level_section():
    """目录问答切片应写入一级目录字段，供考试页目录下拉使用。"""

    outline = [
        {"level": 0, "title": "一、Java 基础", "page": 1},
        {"level": 1, "title": "1、什么是 JVM？", "page": 1},
        {"level": 1, "title": "2、什么是 JDK？", "page": 1},
        {"level": 1, "title": "3、什么是 JRE？", "page": 1},
        {"level": 0, "title": "二、集合", "page": 1},
        {"level": 1, "title": "1、ArrayList 和 LinkedList 有什么区别？", "page": 1},
        {"level": 1, "title": "2、HashMap 的底层结构是什么？", "page": 1},
        {"level": 1, "title": "3、ConcurrentHashMap 如何保证线程安全？", "page": 1},
    ]
    text = "\n".join(
        [
            "1、什么是 JVM？",
            "JVM 是 Java 虚拟机。",
            "2、什么是 JDK？",
            "JDK 是 Java 开发工具包。",
            "3、什么是 JRE？",
            "JRE 是 Java 运行环境。",
            "1、ArrayList 和 LinkedList 有什么区别？",
            "ArrayList 查询快，LinkedList 插入删除更灵活。",
            "2、HashMap 的底层结构是什么？",
            "HashMap 主要由数组、链表和红黑树组成。",
            "3、ConcurrentHashMap 如何保证线程安全？",
            "ConcurrentHashMap 通过分段或节点级并发控制保证线程安全。",
        ]
    )
    documents = [Document(page_content=text, metadata={"page": 0, "_pdf_outline": outline})]

    segments, qa_items = _parser().build_segments_and_qas(
        document_id="doc_1",
        documents=documents,
        document_type="qa",
        split_strategy="outline_qa",
    )

    assert segments
    assert qa_items
    first_metadata = segments[0].metadata
    assert first_metadata["section_title"] == "一、Java 基础"
    assert first_metadata["section_path"] == "一、Java 基础 / 1、什么是 JVM？"
    assert first_metadata["section_first_level"] == "一、Java 基础"
    assert qa_items[0].metadata["section_first_level"] == "一、Java 基础"


def test_exam_section_filter_should_match_first_level_section():
    """考试按一级目录筛选时，应命中该目录下的所有子题。"""

    metadata = {
        "section_path": "一、Java 基础 / 1、什么是 JVM？",
        "section_first_level": "一、Java 基础",
    }

    assert exam_service._match_section(metadata, "一、Java 基础")


def test_outline_qa_index_documents_should_keep_directory_metadata():
    """目录问答写入 Qdrant 前，最终 Document metadata 必须保留目录字段。"""

    outline = [
        {"level": 0, "title": "一、Java 基础", "page": 1},
        {"level": 1, "title": "1、什么是 JVM？", "page": 1},
        {"level": 1, "title": "2、什么是 JDK？", "page": 1},
        {"level": 1, "title": "3、什么是 JRE？", "page": 1},
        {"level": 0, "title": "二、集合", "page": 1},
        {"level": 1, "title": "1、ArrayList 和 LinkedList 有什么区别？", "page": 1},
        {"level": 1, "title": "2、HashMap 的底层结构是什么？", "page": 1},
        {"level": 1, "title": "3、ConcurrentHashMap 如何保证线程安全？", "page": 1},
    ]
    text = "\n".join(
        [
            "1、什么是 JVM？",
            "JVM 是 Java 虚拟机。",
            "2、什么是 JDK？",
            "JDK 是 Java 开发工具包。",
            "3、什么是 JRE？",
            "JRE 是 Java 运行环境。",
            "1、ArrayList 和 LinkedList 有什么区别？",
            "ArrayList 查询快，LinkedList 插入删除更灵活。",
            "2、HashMap 的底层结构是什么？",
            "HashMap 主要由数组、链表和红黑树组成。",
            "3、ConcurrentHashMap 如何保证线程安全？",
            "ConcurrentHashMap 通过分段或节点级并发控制保证线程安全。",
        ]
    )
    service = object.__new__(VectorStoreService)
    service.document_parser = _parser()

    index_documents = service.build_index_documents(
        document_id="doc_1",
        filename="Java面试题.pdf",
        file_md5="md5",
        version=1,
        documents=[Document(page_content=text, metadata={"page": 0, "_pdf_outline": outline})],
        document_type="qa",
        split_strategy="outline_qa",
    )

    first_metadata = index_documents[0].metadata
    assert first_metadata["split_strategy"] == "outline_qa"
    assert first_metadata["section_path"] == "一、Java 基础 / 1、什么是 JVM？"
    assert first_metadata["section_first_level"] == "一、Java 基础"


def test_outline_qa_should_not_silently_fallback_to_numbered_qa():
    """用户手动选择目录问答时，目录不可用应失败，不能静默改成编号问答。"""

    documents = [
        Document(
            page_content="\n".join(
                [
                    "1、什么是 JVM？",
                    "答：JVM 是 Java 虚拟机。",
                    "2、什么是 JDK？",
                    "答：JDK 是 Java 开发工具包。",
                    "3、什么是 JRE？",
                    "答：JRE 是 Java 运行环境。",
                ]
            ),
            metadata={"page": 0},
        )
    ]

    with pytest.raises(ValueError, match="目录问答切分失败"):
        _parser().build_segments_and_qas(
            document_id="doc_without_outline",
            documents=documents,
            document_type="qa",
            split_strategy="outline_qa",
        )
