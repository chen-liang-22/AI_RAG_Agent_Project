import subprocess
import sys

import pytest
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


def test_document_parse_rules_requires_explicit_config():
    """文档解析格式必须来自配置，避免代码里偷偷保留业务格式硬编码。"""

    from core.rag.document_parser import DocumentParseRules

    with pytest.raises(ValueError, match="document_parse_rules"):
        DocumentParseRules.from_config({})


def test_numbered_qa_parser_uses_configured_parse_rules():
    """编号和答案前缀改成新资料格式时，只需要换解析规则配置。"""

    from core.rag.document_parser import DocumentParseRules, DocumentParser

    parser = DocumentParser(
        RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
        parse_rules=DocumentParseRules.from_config(
            {
                "numbered_item_pattern": r"^\s*(\d+)\)\s*(.+)$",
                "heading_pattern": r"^\s*@@\s*(.+)$",
                "number_prefix_pattern": r"^\s*(\d+)\)\s*",
                "number_prefix_only_pattern": r"^\s*\d+\)\s*$",
                "answer_prefix_pattern": r"^(?:回答)[:：]\s*",
                "invalid_answer_pattern": r"^(?:回答)[:：]?$",
                "question_marks": ["？"],
            }
        ),
    )

    _, qa_items = parser.build_segments_and_qas(
        document_id="doc_custom_rules",
        documents=[Document(page_content="@@ 自定义分类\n1) 自定义问题？\n回答：自定义答案。", metadata={})],
        document_type="qa",
        split_strategy="numbered_qa",
    )

    assert qa_items[0].category == "自定义分类"
    assert qa_items[0].question == "自定义问题？"
    assert qa_items[0].answer == "自定义答案。"


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


def test_llm_semantic_strategy_falls_back_to_recursive_when_spans_invalid():
    from core.rag.document_parser import DocumentParser

    documents = [Document(page_content="只有一段短文本。")]
    parser = DocumentParser(
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
        semantic_splitter=_FakeSemanticSplitter(
            [
                {
                    "start": 0,
                    "end": 999,
                    "title": "越界范围",
                    "content_type": "segment",
                }
            ]
        ),
    )

    segments, qa_items = parser.build_segments_and_qas(
        document_id="doc_2",
        documents=documents,
        document_type="text",
        split_strategy="llm_semantic",
    )

    assert [segment.content for segment in segments] == ["只有一段短文本。"]
    assert segments[0].metadata["split_strategy"] == "recursive"
    assert qa_items == []


def test_recommendation_normalization_accepts_llm_semantic(monkeypatch):
    from app_v2.application.knowledge import upload_preview_service

    monkeypatch.setattr(upload_preview_service, "DictionaryRepository", _FakeDictionaryRepository)

    recommendation = upload_preview_service._normalize_recommendation(
        {
            "document_type": "qa",
            "split_strategy": "llm_semantic",
            "confidence": 0.82,
            "reasons": ["模型认为语义边界比固定编号更可靠"],
        }
    )

    assert recommendation["document_type"] == "text"
    assert recommendation["split_strategy"] == "llm_semantic"
    assert recommendation["confidence"] == 0.82
    assert recommendation["reasons"] == ["模型认为语义边界比固定编号更可靠"]


class _FakeSemanticSplitter:
    def __init__(self, plans):
        self.plans = plans

    def split(self, documents):
        return self.plans


class _FakeDictionaryRepository:
    def list_enabled_codes(self, dictionary_code):
        if dictionary_code == "document_structure":
            return ["text", "qa", "numbered"]
        if dictionary_code == "split_strategy":
            return ["recursive", "numbered_qa", "outline_qa", "numbered_segments", "llm_semantic"]
        return []
