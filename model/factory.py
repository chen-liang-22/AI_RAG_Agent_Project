from abc import ABC, abstractmethod  # 定义抽象工厂基类，约束子类必须实现 generator
from typing import Optional  # 标注 generator 可能返回模型对象，也可能返回 None
from langchain_core.embeddings import Embeddings  # 向量模型的通用类型
from langchain_community.chat_models.tongyi import BaseChatModel  # 聊天模型的通用类型
from langchain_community.embeddings import DashScopeEmbeddings  # 通义 DashScope embedding 模型
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义聊天模型
from utils.config_handler import rag_conf  # 读取 config 下的 RAG/模型配置


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass  # 抽象方法，具体子类负责创建聊天模型或向量模型


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # streaming=True 是真实流式输出的模型层开关。
        #
        # 如果这里不开启：
        # - Agent 的 stream 接口仍然可以被调用；
        # - 但底层模型不会边生成边吐 token；
        # - 前端最终看到的仍可能接近“一次性返回”。
        #
        # 开启后，ChatTongyi 会把模型生成过程拆成多个 AIMessageChunk，
        # ReactAgent.execute_stream() 才能逐段 yield 给 FastAPI 的 SSE 接口。
        return ChatTongyi(model=rag_conf["chat_model_name"], streaming=True)  # 创建聊天模型，并打开真实流式输出


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])  # 创建 embedding 模型，用于文档向量化和检索


chat_model = ChatModelFactory().generator()  # 全局聊天模型实例，Agent 会复用它
embed_model = EmbeddingsFactory().generator()  # 全局向量模型实例，RAG 向量库会复用它
