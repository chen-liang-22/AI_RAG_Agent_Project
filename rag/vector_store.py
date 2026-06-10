import os  # 用于判断 MD5 记录文件是否存在

from langchain_core.documents import Document  # LangChain 的文档对象，包含 page_content 和 metadata
from langchain_qdrant import QdrantVectorStore  # LangChain 对 Qdrant 的向量库封装
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归文本切分器
from qdrant_client import QdrantClient, models  # QdrantClient 用于按 document_id 删除向量

from model.factory import embed_model  # embedding 模型，用于把文本分片转成向量
from rag.document_parser import DocumentParser  # 通用文档解析器，负责识别、切分和 FAQ 抽取
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

    def __init__(self, *, force_recreate: bool | None = None):
        """初始化向量库连接和文本切分器。

        这里只是准备好 QdrantVectorStore 和 splitter。
        真正把文件写入向量库是在 load_document() 方法里完成的。
        """

        recreate_collection = qdrant_conf.get("force_recreate", False) if force_recreate is None else force_recreate

        self.vector_store = QdrantVectorStore.construct_instance(
            embedding=embed_model,  # 文本转向量时使用的 embedding 模型
            collection_name=get_qdrant_collection_name(),  # Qdrant collection 名称
            client_options=get_qdrant_client_options(),  # Qdrant 连接配置
            distance=get_qdrant_distance(),  # 向量相似度距离算法
            force_recreate=recreate_collection,  # 是否强制重建 collection
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=qdrant_conf["chunk_size"],  # 每个文本分片的目标长度
            chunk_overlap=qdrant_conf["chunk_overlap"],  # 相邻分片重叠长度，避免语义被切断
            separators=qdrant_conf["separators"],  # 切分优先级，优先按段落/换行/标点切
            length_function=len,  # 用 Python len 计算文本长度
        )
        self.document_parser = DocumentParser(self.spliter)  # 通用文档解析器；支持上传预览、通用片段和 FAQ 抽取

    @classmethod
    def recreate_collection_service(cls) -> "VectorStoreService":
        """删除并重建当前 Qdrant collection，然后返回新的向量库服务实例。

        这个方法只给“全量重建索引”使用。
        它可以清理历史遗留的旧 points，尤其是没有 document_id 的旧数据。
        """

        return cls(force_recreate=True)

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

    @staticmethod
    def build_metadata_filter(filters: dict[str, list[str]] | None) -> models.Filter | None:
        """把业务过滤条件转换成 Qdrant Filter。

        filters 的结构来自 RuleBasedIntentAnalyzer，例如：

            {
                "unit_type": ["guide", "faq"],
                "category": ["选购指南", "常见问答"]
            }

        LangChain 写入 Qdrant 时，Document.metadata 会被放在 payload.metadata 下面。
        所以过滤字段要写成：
        - metadata.unit_type
        - metadata.category
        """

        if not filters:
            return None

        conditions: list[models.FieldCondition] = []

        for key, values in filters.items():
            clean_values = [value for value in values if value]
            if not clean_values:
                continue

            conditions.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchAny(any=clean_values),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    def search_documents(
            self,
            query: str,
            *,
            k: int | None = None,
            filters: dict[str, list[str]] | None = None,
    ) -> list[Document]:
        """检索 Qdrant 文档，并保留向量相似度分数。

        这个方法用于新版多路召回流程。
        与 get_retriever() 的区别：
        - 可以传 metadata filter。
        - 可以拿到 similarity score。
        - 会把 score 写回 Document.metadata["_vector_score"]，方便 rerank 使用。
        """

        if not query.strip():
            return []

        search_filter = self.build_metadata_filter(filters)
        result_limit = k or qdrant_conf["k"]
        results = self.vector_store.similarity_search_with_score(
            query,
            k=result_limit,
            filter=search_filter,
        )

        documents: list[Document] = []
        for document, score in results:
            metadata = dict(document.metadata)
            metadata["_vector_score"] = float(score)
            documents.append(Document(page_content=document.page_content, metadata=metadata))

        return documents

    def search_faq_documents(self, query: str, *, k: int = 8) -> list[Document]:
        """只在 Qdrant 的 FAQ 向量里做语义检索。"""

        return self.search_documents(
            query,
            k=k,
            filters={"content_type": ["faq"]},
        )

    @staticmethod
    def scroll_faq_documents(
            *,
            question_no: int | None = None,
            limit: int = 1000,
    ) -> list[Document]:
        """从 Qdrant payload 中滚动读取 FAQ points。

        这个方法用于“第95问是什么”“100问都有哪些”这类结构化查询。
        它不走 SQLite，直接使用 Qdrant payload filter。
        """

        conditions: list[models.FieldCondition] = [
            models.FieldCondition(
                key="metadata.content_type",
                match=models.MatchValue(value="faq"),
            )
        ]
        if question_no is not None:
            conditions.append(
                models.FieldCondition(
                    key="metadata.question_no",
                    match=models.MatchValue(value=question_no),
                )
            )

        client = QdrantClient(**get_qdrant_client_options())
        points, _ = client.scroll(
            collection_name=get_qdrant_collection_name(),
            scroll_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        documents: list[Document] = []
        for point in points:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata") or {})
            metadata["_point_id"] = str(point.id)
            documents.append(
                Document(
                    page_content=str(payload.get("page_content") or ""),
                    metadata=metadata,
                )
            )

        return documents

    def preview_file(
            self,
            *,
            filename: str,
            file_path: str,
            sample_limit: int = 5000,
    ) -> dict:
        """读取文件样本文本并识别文档类型。

        这个方法用于上传预览接口，不写 SQLite，也不写 Qdrant。
        """

        documents = self.get_file_documents(file_path)
        sample_text = "\n\n".join(document.page_content for document in documents)[:sample_limit]
        detection = self.document_parser.detect_document_type(filename, sample_text)
        return {
            "filename": filename,
            "document_type": detection.document_type,
            "split_strategy": detection.split_strategy,
            "confidence": detection.confidence,
            "reasons": detection.reasons,
            "llm_used": detection.llm_used,
            "sample_text": sample_text,
        }

    @staticmethod
    def get_file_documents(read_path: str) -> list[Document]:
        """根据文件后缀选择 loader，把单个文件读取成 Document 列表。"""

        lower_read_path = read_path.lower()

        if lower_read_path.endswith(".txt"):
            return txt_loader(read_path)

        if lower_read_path.endswith(".pdf"):
            return pdf_loader(read_path)

        return []

    def build_index_documents(
            self,
            *,
            document_id: str,
            filename: str,
            file_md5: str,
            version: int,
            documents: list[Document],
            document_type: str,
            split_strategy: str,
    ) -> tuple[list[Document], list[dict], list[dict]]:
        """把原始 Document 切分成 Qdrant 文档、segments 和 FAQ。

        返回：
        - index_documents：写入 Qdrant 的 Document 列表。
        - segments：写入 SQLite document_segments 的通用片段。
        - faq_items：写入 SQLite faq_items 的结构化问答。
        """

        segments, faq_items = self.document_parser.build_segments_and_faqs(
            document_id=document_id,
            documents=documents,
            document_type=document_type,
            split_strategy=split_strategy,
        )
        faq_by_segment_id = {item.segment_id: item for item in faq_items}
        index_documents: list[Document] = []

        for segment in segments:
            faq = faq_by_segment_id.get(segment.segment_id)
            metadata = {
                "document_id": document_id,
                "segment_id": segment.segment_id,
                "chunk_id": segment.segment_id,
                "content_type": "faq" if faq else "segment",
                "document_type": document_type,
                "unit_type": document_type,
                "split_strategy": split_strategy,
                "source_file": filename,
                "file_md5": file_md5,
                "version": version,
                "segment_index": segment.segment_index,
                "chunk_index": segment.segment_index,
                "page_no": segment.page_no,
                "source_page": segment.page_no,
                "heading_path": segment.heading_path,
                "question_no": faq.question_no if faq else segment.metadata.get("question_no"),
                "faq_id": faq.faq_id if faq else None,
                "question": faq.question if faq else None,
                "category": faq.category if faq else segment.heading_path,
            }

            index_documents.append(Document(page_content=segment.content, metadata=metadata))

        segment_dicts = [
            {
                "segment_id": segment.segment_id,
                "segment_index": segment.segment_index,
                "content": segment.content,
                "content_hash": segment.content_hash,
                "page_no": segment.page_no,
                "heading_path": segment.heading_path,
                "metadata": segment.metadata,
            }
            for segment in segments
        ]
        faq_dicts = [
            {
                "faq_id": item.faq_id,
                "segment_id": item.segment_id,
                "question_no": item.question_no,
                "question": item.question,
                "answer": item.answer,
                "category": item.category,
                "tags": item.tags,
                "metadata": item.metadata,
            }
            for item in faq_items
        ]

        return index_documents, segment_dicts, faq_dicts

    @staticmethod
    def delete_document_vectors(document_id: str) -> None:
        """按 document_id 删除 Qdrant 中属于某个文件的所有向量。"""

        client = QdrantClient(**get_qdrant_client_options())
        client.delete(
            collection_name=get_qdrant_collection_name(),
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def index_file(
            self,
            document: dict,
            *,
            document_type: str | None = None,
            split_strategy: str | None = None,
    ) -> tuple[int, list[dict], list[dict]]:
        """把 documents 表中的单个文件重新写入 Qdrant。

        这个方法用于上传、重新索引等新流程。
        它会先删除该 document_id 的旧向量，再写入新向量。
        """

        file_path = document["file_path"]
        document_id = document["document_id"]
        filename = document["filename"]
        file_md5 = document["file_md5"]
        version = int(document["version"])

        documents = self.get_file_documents(file_path)
        if not documents:
            raise ValueError(f"文件 {file_path} 没有有效文本内容")

        if not document_type or not split_strategy:
            preview = self.preview_file(filename=filename, file_path=file_path)
            document_type = document_type or preview["document_type"]
            split_strategy = split_strategy or preview["split_strategy"]

        index_documents, segments, faq_items = self.build_index_documents(
            document_id=document_id,
            filename=filename,
            file_md5=file_md5,
            version=version,
            documents=documents,
            document_type=document_type,
            split_strategy=split_strategy,
        )

        if not index_documents:
            raise ValueError(f"文件 {file_path} 分片后没有有效文本内容")

        self.delete_document_vectors(document_id)
        self.vector_store.add_documents(index_documents)

        return len(index_documents), segments, faq_items

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


