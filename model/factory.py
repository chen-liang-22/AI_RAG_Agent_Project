import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from rag.knowledge_store import KnowledgeStore
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self, model_name: str | None = None) -> Optional[Embeddings | BaseChatModel]:
        provider = str(rag_conf.get("chat_provider") or "tongyi").strip().lower()
        selected_model_name = model_name or rag_conf["chat_model_name"]

        if provider in {"openai_compatible", "openai-compatible", "openai"}:
            return ChatOpenAI(
                model=selected_model_name,
                base_url=rag_conf.get("openai_base_url"),
                api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "empty",
                streaming=True,
            )

        if provider == "tongyi":
            return ChatTongyi(model=selected_model_name, streaming=True)

        raise ValueError(f"不支持的 chat_provider：{provider}")


def normalize_chat_model_mode(model_mode: str | None) -> str:
    """从模型档位字典里归一化前端传入的模型模式。"""

    store = KnowledgeStore()
    return store.normalize_dictionary_code("model_mode", model_mode)


def get_chat_model_name_for_mode(model_mode: str | None = None) -> str:
    mode = normalize_chat_model_mode(model_mode)
    return str(rag_conf.get(f"chat_model_{mode}") or rag_conf["chat_model_name"]).strip()


@lru_cache(maxsize=8)
def _cached_chat_model(model_name: str) -> Optional[Embeddings | BaseChatModel]:
    return ChatModelFactory().generator(model_name=model_name)


def get_chat_model(model_mode: str | None = None) -> Optional[Embeddings | BaseChatModel]:
    return _cached_chat_model(get_chat_model_name_for_mode(model_mode))


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = get_chat_model()
embed_model = EmbeddingsFactory().generator()
