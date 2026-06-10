import os  # 用于判断 MD5 记录文件是否存在

from langchain_core.documents import Document  # LangChain 的文档对象，包含 page_content 和 metadata
from langchain_qdrant import QdrantVectorStore  # LangChain 对 Qdrant 的向量库封装
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归文本切分器

from model.factory import embed_model  # embedding 模型，用于把文本分片转成向量
from utils.config_handler import qdrant_conf  # 读取 config/qdrant.yml 中的向量库配置
from utils.file_handler import (  # 文件处理工具
    get_file_md5_hex,  # 计算文件 MD5，用于判断文件是否已处理过
    listdir_with_allowed_type,  # 列出允许类型的知识库文件
    pdf_loader,  # PDF 加载器
    txt_loader,  # TXT 加载器
)
from utils.logger_handler import logger  # 项目统一日志
from utils.path_tool import get_abs_path  # 把相对路径转换成项目绝对路径
from utils.qdrant_options import (
    get_qdrant_client_options,  # 读取 Qdrant 连接参数，如 url/host/port/grpc_port
    get_qdrant_collection_name,  # 读取当前 collection 名称
    get_qdrant_distance,  # 读取向量距离算法，如 COSINE
)


class VectorStoreService:
    """Qdrant 向量库服务。

    这个类主要负责三件事：

    1. 初始化 QdrantVectorStore
       - 连接 Qdrant
       - 指定 collection
       - 指定 embedding 模型
       - 指定向量距离算法

    2. 把本地知识库文件写入向量库
       - 扫描 data 目录
       - 只读取 txt/pdf
       - 计算文件 MD5 做去重
       - 读取文件为 Document
       - 切分 Document
       - 调用 embedding
       - 写入 Qdrant

    3. 提供 retriever 给 RAG 使用
       - 用户问题会先转成向量
       - Qdrant 按相似度召回 topK 文本分片

    注意：
    - Qdrant 里不是保存“完整文件”，而是保存“文本分片 + 向量 + metadata”。
    - MD5 只能避免重复加载同一份文件，不能自动删除旧版本分片。
    """

    def __init__(self):
        """初始化向量库连接和文本切分器。

        这里只是准备好 QdrantVectorStore 和 splitter。
        真正把文件写入向量库是在 load_document() 方法里完成的。
        """

        self.vector_store = QdrantVectorStore.construct_instance(
            embedding=embed_model,  # 文本转向量时使用的 embedding 模型
            collection_name=get_qdrant_collection_name(),  # Qdrant collection 名称
            client_options=get_qdrant_client_options(),  # Qdrant 连接配置
            distance=get_qdrant_distance(),  # 向量相似度距离算法
            force_recreate=qdrant_conf.get("force_recreate", False),  # 是否强制重建 collection
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=qdrant_conf["chunk_size"],  # 每个文本分片的目标长度
            chunk_overlap=qdrant_conf["chunk_overlap"],  # 相邻分片重叠长度，避免语义被切断
            separators=qdrant_conf["separators"],  # 切分优先级，优先按段落/换行/标点切
            length_function=len,  # 用 Python len 计算文本长度
        )

    def get_retriever(self):
        """获取 LangChain retriever。

        retriever 是 RAG 检索阶段用的对象。

        调用方式大概是：

            docs = retriever.invoke("用户问题")

        它内部会：
        1. 把用户问题转成 embedding 向量。
        2. 到 Qdrant collection 里做相似度检索。
        3. 返回最相似的 k 个 Document。
        """

        return self.vector_store.as_retriever(search_kwargs={"k": qdrant_conf["k"]})

    def load_document(self):
        """
        从数据文件夹读取知识库文件，并写入 Qdrant 向量库。

        整体流程：

            data 文件
              ↓
            loader 读取成 Document
              ↓
            splitter 切分成多个小 Document
              ↓
            embedding 模型生成向量
              ↓
            Qdrant 保存 point

        每个 Qdrant point 大致包含：

            vector: embedding 向量
            payload.page_content: 文本分片内容
            payload.metadata: 文件来源、页码等元数据

        这里会用文件 MD5 做简单去重：
        - 如果 MD5 已存在 qdrant_md5.text，说明这个文件处理过，跳过。
        - 如果 MD5 不存在，加载并写入 Qdrant，之后记录 MD5。

        重要限制：
        - 如果文件内容变了，会产生新的 MD5，于是会新增一批分片。
        - 旧分片不会自动删除。
        - 如果要真正更新文件，建议先清理 collection 或做 document_id 级删除。

        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            """判断某个文件 MD5 是否已经被处理过。

            返回：
            - True：这个 MD5 已经在记录文件里，说明文件已经加载过。
            - False：这个 MD5 没出现过，说明文件需要加载。
            """

            if not os.path.exists(get_abs_path(qdrant_conf["md5_hex_store"])):
                # 如果 MD5 记录文件不存在，先创建一个空文件。
                # 这通常发生在第一次加载知识库时。
                open(get_abs_path(qdrant_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False  # 记录文件刚创建，当前 MD5 肯定没处理过

            with open(get_abs_path(qdrant_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                # 逐行读取历史 MD5。
                # 文件很小时这样写没问题；如果以后文件很多，可以改成 set 加速。
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  # 找到相同 MD5，说明处理过

                return False  # 遍历完没找到，说明没处理过

        def save_md5_hex(md5_for_check: str):
            """把已经成功写入向量库的文件 MD5 追加到记录文件。"""

            with open(get_abs_path(qdrant_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")  # 一行保存一个 MD5

        def get_file_documents(read_path: str):
            """根据文件后缀选择对应 loader，把文件读取成 Document 列表。"""

            if read_path.endswith("txt"):
                # txt_loader 会返回一个或多个 Document。
                # Document.page_content 是文本内容，metadata.source 是文件路径。
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                # pdf_loader 通常会按页读取 PDF。
                # 每页会带 page、source 等 metadata。
                return pdf_loader(read_path)

            # 不支持的文件类型返回空列表。
            # 实际上外层已经按 allow_knowledge_file_type 过滤过，这里是兜底。
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(qdrant_conf["data_path"]),  # 知识库目录，默认 data
            tuple(qdrant_conf["allow_knowledge_file_type"]),  # 允许的文件类型，如 txt/pdf
        )

        for path in allowed_files_path:
            # 计算文件 MD5。
            # 这里的 MD5 是按文件二进制内容计算，不是按文件名计算。
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                # 第一步：把文件读取成 LangChain Document。
                # TXT 通常是一个 Document；PDF 通常是一页一个 Document。
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                # 第二步：把完整 Document 切成多个小 Document。
                # 每个小 Document 会保留原始 metadata，并拥有自己的 page_content。
                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 第三步：写入 Qdrant。
                # add_documents 内部会：
                # 1. 提取每个 Document 的 page_content。
                # 2. 调用 embed_model 生成向量。
                # 3. 把向量和 metadata 一起写入 collection。
                self.vector_store.add_documents(split_document)

                # 只有 add_documents 成功后，才记录 MD5。
                # 这样如果中途失败，下次还能继续尝试加载。
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info=True 会记录完整异常堆栈，方便定位是读取、切分、embedding 还是 Qdrant 写入失败。
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    # 直接运行本文件时，做一个简单的手动测试。
    vs = VectorStoreService()

    # 加载 data 目录中的知识库文件。
    vs.load_document()

    # 获取检索器。
    retriever = vs.get_retriever()

    # 用“迷路”做一次相似度检索测试。
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)


