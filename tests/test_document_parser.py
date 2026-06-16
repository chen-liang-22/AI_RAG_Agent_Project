from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.document_parser import DocumentParser


def test_qa_parser_splits_numbered_questions_and_answers():
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
            metadata={"page": 1, "source": "qa.txt"},
        )
    ]

    segments, qa_items = parser.build_segments_and_qas(
        document_id="doc_test",
        documents=documents,
        document_type="qa",
        split_strategy="numbered_qa",
    )

    assert len(segments) == 2
    assert len(qa_items) == 2
    assert qa_items[0].question == "扫拖一体机器人适合小户型吗？"
    assert qa_items[0].answer == "适合，重点看避障和水箱容量。"
    assert qa_items[0].category == "拖扫功能融合类"
    assert segments[0].metadata["document_type"] == "qa"
    assert segments[0].metadata["split_strategy"] == "numbered_qa"


def test_document_type_detection_identifies_numbered_text():
    parser = DocumentParser(
        RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", "。", " ", ""],
        )
    )

    detection = parser.detect_document_type(
        "编号条目.txt",
        "\n".join(
            [
                "1. 机器人迷路：检查传感器并清理机身。",
                "2. 无法回充：确认充电座位置和电源。",
                "3. 吸力下降：清理尘盒和滤网。",
            ]
        ),
    )

    assert detection.document_type == "numbered"
    assert detection.split_strategy == "numbered_segments"


def test_document_type_detection_identifies_outline_qa():
    parser = DocumentParser(
        RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", "。", " ", ""],
        )
    )
    outline = [
        {"level": 0, "title": "Java集合面试题", "page": 1},
        {"level": 1, "title": "1、ArrayList和LinkedList的区别", "page": 1},
        {"level": 1, "title": "2、HashMap和Hashtable的区别", "page": 1},
        {"level": 1, "title": "3、ConcurrentHashMap的原理", "page": 2},
        {"level": 0, "title": "JVM面试题", "page": 3},
        {"level": 1, "title": "1、什么是类加载器？", "page": 3},
        {"level": 1, "title": "2、什么是双亲委派？", "page": 3},
        {"level": 1, "title": "3、GC Roots有哪些？", "page": 4},
    ]

    detection = parser.detect_document_type(
        "Java面试题.pdf",
        "Java集合面试题\n1、ArrayList和LinkedList的区别\n答案内容",
        outline=outline,
    )

    assert detection.document_type == "qa"
    assert detection.split_strategy == "outline_qa"


def test_outline_qa_parser_splits_by_pdf_outline():
    parser = DocumentParser(
        RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", "。", " ", ""],
        )
    )
    outline = [
        {"level": 0, "title": "Java集合面试题", "page": 1},
        {"level": 1, "title": "1、ArrayList和LinkedList的区别", "page": 1},
        {"level": 1, "title": "2、HashMap和Hashtable的区别", "page": 1},
        {"level": 1, "title": "3、ConcurrentHashMap的原理", "page": 2},
        {"level": 0, "title": "JVM面试题", "page": 3},
        {"level": 1, "title": "1、什么是类加载器？", "page": 3},
        {"level": 1, "title": "2、什么是双亲委派？", "page": 3},
        {"level": 1, "title": "3、GC Roots有哪些？", "page": 4},
    ]
    documents = [
        Document(
            page_content=(
                "Java集合面试题\n"
                "1、ArrayList和LinkedList的区别\n"
                "ArrayList 基于数组，查询快；LinkedList 基于链表，增删更灵活。\n"
                "2、HashMap和Hashtable的区别\n"
                "HashMap 允许 null，Hashtable 线程安全但较旧。\n"
                "3、ConcurrentHashMap的原理\n"
                "它通过分段或 CAS 降低锁竞争。"
            ),
            metadata={"page": 0, "source": "Java面试题.pdf", "_pdf_outline": outline},
        ),
        Document(
            page_content=(
                "JVM面试题\n"
                "1、什么是类加载器？\n"
                "类加载器负责加载 class 文件。\n"
                "2、什么是双亲委派？\n"
                "优先委托父加载器加载类。\n"
                "3、GC Roots有哪些？\n"
                "包括栈帧引用、静态变量引用等。"
            ),
            metadata={"page": 2, "source": "Java面试题.pdf"},
        ),
    ]

    segments, qa_items = parser.build_segments_and_qas(
        document_id="doc_java",
        documents=documents,
        document_type="qa",
        split_strategy="outline_qa",
    )

    assert len(segments) == 6
    assert len(qa_items) == 6
    assert qa_items[0].category == "Java集合面试题"
    assert qa_items[0].question_no == 1
    assert qa_items[0].question == "ArrayList和LinkedList的区别"
    assert "ArrayList 基于数组" in segments[0].content
    assert segments[0].metadata["split_strategy"] == "outline_qa"
    assert segments[0].metadata["section_path"] == "Java集合面试题 / 1、ArrayList和LinkedList的区别"
    assert segments[3].metadata["section_title"] == "JVM面试题"
