"""模型工厂。

这个模块集中创建聊天模型和 Embedding 模型。
业务代码不要直接 new ChatTongyi / ChatOpenAI，统一通过这里读取配置和字典档位，
这样后续切换 qwen3-max、qwen3.7-max 或 OpenAI compatible 接口时，只改配置和工厂逻辑。
"""

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from app.infrastructure.repositories.dictionary_repository import DictionaryRepository
from core.utils.config_handler import rag_conf
from core.utils.logger_handler import logger


class BaseModelFactory(ABC):
    """模型工厂抽象基类。

    这里使用工厂方法模式：调用方只关心拿到“可调用的模型对象”，
    具体是通义、OpenAI compatible 还是别的实现，由子类决定。
    """

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """创建具体模型实例。"""

        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型工厂。

    根据 `config/app.yml` 中 rag.chat_provider 和 rag.chat_model_name 创建最终回答模型。
    """

    def generator(self, model_name: str | None = None) -> Optional[Embeddings | BaseChatModel]:
        """创建聊天模型实例。

        重要说明：
        - openai_compatible 用于阿里 DashScope 兼容 OpenAI 接口；
        - tongyi 用于 LangChain 的 ChatTongyi；
        - 不在日志里打印 API Key，避免泄露密钥。
        """

        provider = str(rag_conf.get("chat_provider") or "tongyi").strip().lower()
        selected_model_name = model_name or rag_conf["chat_model_name"]
        logger.info("[模型工厂] 创建聊天模型 Provider=%s 模型名称=%s", provider, selected_model_name)

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

    return DictionaryRepository().normalize_code("model_mode", model_mode)


def get_chat_model_name_for_mode(model_mode: str | None = None) -> str:
    """根据模型档位读取具体聊天模型名称。"""

    mode = normalize_chat_model_mode(model_mode)
    return str(rag_conf.get(f"chat_model_{mode}") or rag_conf["chat_model_name"]).strip()


@lru_cache(maxsize=8)
def _cached_chat_model(model_name: str) -> Optional[Embeddings | BaseChatModel]:
    """缓存聊天模型实例，避免每轮对话重复创建 SDK 客户端。"""

    return ChatModelFactory().generator(model_name=model_name)


def get_chat_model(model_mode: str | None = None) -> Optional[Embeddings | BaseChatModel]:
    """获取指定档位的聊天模型。"""

    return _cached_chat_model(get_chat_model_name_for_mode(model_mode))


class EmbeddingsFactory(BaseModelFactory):
    """Embedding 模型工厂。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """创建向量化模型实例。"""

        logger.info("[模型工厂] 创建Embedding模型 模型名称=%s", rag_conf["embedding_model_name"])
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = get_chat_model()
embed_model = EmbeddingsFactory().generator()
