import subprocess
import sys

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def test_document_parser_import_does_not_load_model_factory():
    """普通解析器导入不应提前加载模型工厂配置。"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from core.rag.document_parser import DocumentParser; "
                "print('model.factory' in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_detect_document_type_defaults_to_plain_recursive_without_regex_guessing():
    from core.rag.document_parser import DocumentParser

    parser = DocumentParser(
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
        semantic_splitter=_FakeSemanticSplitter([]),
    )

    detection = parser.detect_document_type(
        "qa.txt",
        "1. 什么是语义切分？\n答：由模型判断边界。",
    )

    assert detection.document_type == "text"
    assert detection.split_strategy == "recursive"
    assert detection.llm_used is False


def test_llm_semantic_strategy_slices_original_text_by_model_spans():
    from core.rag.document_parser import DocumentParser

    documents = [
        Document(page_content="第一段保留原文。"),
        Document(page_content="第二段也必须按原文截取。"),
    ]
    full_text = "第一段保留原文。\n\n第二段也必须按原文截取。"
    splitter = _FakeSemanticSplitter(
        [
            {
                "start": 0,
                "end": len("第一段保留原文。"),
                "title": "第一段",
                "content_type": "segment",
                "reason": "首段完整语义",
            },
            {
                "start": full_text.index("第二段"),
                "end": len(full_text),
                "title": "第二段",
                "content_type": "qa",
                "question": "第二段说了什么？",
                "category": "示例",
                "reason": "问答式语义",
            },
        ]
    )
    parser = DocumentParser(
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
        semantic_splitter=splitter,
    )

    segments, qa_items = parser.build_segments_and_qas(
        document_id="doc_1",
        documents=documents,
        document_type="text",
        split_strategy="llm_semantic",
    )

    assert [segment.content for segment in segments] == [
        "第一段保留原文。",
        "第二段也必须按原文截取。",
    ]
    assert segments[1].metadata["source_start"] == full_text.index("第二段")
    assert segments[1].metadata["source_end"] == len(full_text)
    assert qa_items[0].question == "第二段说了什么？"
    assert qa_items[0].answer == "第二段也必须按原文截取。"


class _FakeSemanticSplitter:
    def __init__(self, plans):
        self.plans = plans

    def split(self, documents):
        return self.plans
