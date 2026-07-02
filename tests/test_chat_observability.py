"""聊天链路可观测性测试。"""

import json

import httpx
from openai import APIConnectionError

from app.application import chat_generation_service


class FakeKnowledgeAnswerService:
    """测试用知识直答服务，记录 trace_id 是否被透传。"""

    def __init__(self):
        self.calls = []

    def stream_answer(self, message: str, **kwargs):
        """模拟流式回答。"""

        self.calls.append((message, kwargs))
        yield "你好"


class FailingAgent:
    """测试用失败 Agent，模拟模型网关连接失败。"""

    def execute_stream(self, *args, **kwargs):
        """模拟 OpenAI SDK 在流式模型调用阶段抛连接异常。"""

        request = httpx.Request("POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        raise APIConnectionError(request=request)
        yield  # pragma: no cover


class FailingKnowledgeAnswerService:
    """测试用失败知识直答服务，模拟模型网关连接失败。"""

    def stream_answer(self, *args, **kwargs):
        """模拟知识直答流式生成阶段抛连接异常。"""

        request = httpx.Request("POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        raise APIConnectionError(request=request)
        yield  # pragma: no cover


def _event_payload(events: list[str], event_name: str) -> dict:
    """从 SSE 事件列表中读取指定事件 payload。"""

    prefix = f"event: {event_name}\ndata: "
    for event in events:
        if event.startswith(prefix):
            return json.loads(event[len(prefix):].strip())
    raise AssertionError(f"缺少 SSE 事件：{event_name}")


def test_direct_rag_stream_includes_and_forwards_trace_id(monkeypatch):
    """知识直答流式输出应在 SSE 和 RAG 调用中透传 trace_id。"""

    fake_service = FakeKnowledgeAnswerService()
    monkeypatch.setattr(chat_generation_service, "_get_knowledge_answer_service", lambda: fake_service)
    monkeypatch.setattr(chat_generation_service, "normalize_chat_model_mode", lambda model_mode: model_mode or "medium")
    monkeypatch.setattr(chat_generation_service, "get_chat_model_name_for_mode", lambda model_mode: "test-model")
    monkeypatch.setattr(chat_generation_service, "normalize_qdrant_collection_name", lambda collection_name: collection_name or "agent")
    monkeypatch.setattr(chat_generation_service, "_save_chat_exchange", lambda **kwargs: None)

    events = list(chat_generation_service._stream_direct_rag(
        "扫地机器人怎么保养",
        user_id="user_1",
        conversation_id="conv_1",
        history=[],
        model_mode="medium",
        collection_name="agent",
        trace_id="chat_trace_1",
    ))

    assert _event_payload(events, "meta")["trace_id"] == "chat_trace_1"
    assert _event_payload(events, "done")["trace_id"] == "chat_trace_1"
    assert fake_service.calls[0][1]["trace_id"] == "chat_trace_1"


def test_agent_stream_returns_sse_error_when_model_connection_fails(monkeypatch):
    """Agent 流式模型连接失败时应返回 SSE error，不能让异常穿透 ASGI。"""

    monkeypatch.setattr(chat_generation_service, "_get_agent", lambda: FailingAgent())
    monkeypatch.setattr(chat_generation_service, "normalize_chat_model_mode", lambda model_mode: "high")
    monkeypatch.setattr(chat_generation_service, "get_chat_model_name_for_mode", lambda model_mode: "test-model")

    events = list(chat_generation_service._stream_agent(
        "你好",
        user_id="user_1",
        conversation_id="conv_1",
        history=[],
        trace_id="chat_trace_error",
    ))

    assert _event_payload(events, "meta")["trace_id"] == "chat_trace_error"
    assert _event_payload(events, "error")["error"] == "模型服务连接失败，请检查模型网关、代理或网络配置"


def test_direct_rag_stream_returns_sse_error_when_model_connection_fails(monkeypatch):
    """知识直答流式模型连接失败时也应返回 SSE error。"""

    monkeypatch.setattr(chat_generation_service, "_get_knowledge_answer_service", lambda: FailingKnowledgeAnswerService())
    monkeypatch.setattr(chat_generation_service, "normalize_chat_model_mode", lambda model_mode: model_mode or "medium")
    monkeypatch.setattr(chat_generation_service, "get_chat_model_name_for_mode", lambda model_mode: "test-model")
    monkeypatch.setattr(chat_generation_service, "normalize_qdrant_collection_name", lambda collection_name: collection_name or "agent")

    events = list(chat_generation_service._stream_direct_rag(
        "你好",
        user_id="user_1",
        conversation_id="conv_1",
        history=[],
        model_mode="medium",
        collection_name="agent",
        trace_id="chat_trace_direct_error",
    ))

    assert _event_payload(events, "meta")["trace_id"] == "chat_trace_direct_error"
    assert _event_payload(events, "error")["error"] == "模型服务连接失败，请检查模型网关、代理或网络配置"
