from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.document_parser import DocumentParser
from rag.file_processors.factory import FileProcessorFactory
from rag.file_processors.pdf_processor import PdfFileProcessor
from rag.file_processors.txt_processor import TxtFileProcessor
from rag.split_strategies.factory import SplitStrategyFactory
from rag.split_strategies.numbered_qa_strategy import NumberedQaSplitStrategy
from rag.split_strategies.recursive_strategy import RecursiveSplitStrategy


def test_file_processor_factory_selects_registered_processors():
    """文件处理器工厂应根据文件类型选择对应策略实现。"""

    assert isinstance(FileProcessorFactory.get_processor("txt"), TxtFileProcessor)
    assert isinstance(FileProcessorFactory.get_processor(".pdf"), PdfFileProcessor)


def test_split_strategy_factory_selects_registered_strategies():
    """切片策略工厂应根据 split_strategy 选择对应策略实现。"""

    assert isinstance(SplitStrategyFactory.get_strategy("numbered_qa"), NumberedQaSplitStrategy)
    assert isinstance(SplitStrategyFactory.get_strategy("unknown"), RecursiveSplitStrategy)


def test_document_parser_uses_split_strategy_factory_for_numbered_qa():
    """DocumentParser 应通过切片策略工厂完成编号问答切分。"""

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
                "### 基础问答\n"
                "1. 什么是 JVM？\n"
                "- 答：JVM 是运行 Java 字节码的虚拟机。"
            ),
            metadata={"page": 1, "source": "qa.txt"},
        )
    ]

    segments, qa_items = parser.build_segments_and_qas(
        document_id="doc_factory",
        documents=documents,
        document_type="qa",
        split_strategy="numbered_qa",
    )

    assert len(segments) == 1
    assert len(qa_items) == 1
    assert segments[0].metadata["split_strategy"] == "numbered_qa"
    assert qa_items[0].question == "什么是 JVM？"
