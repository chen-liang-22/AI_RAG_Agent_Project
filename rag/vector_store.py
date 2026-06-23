from typing import Any

from langchain_core.documents import Document  # LangChain 的文档对象，包含 page_content 和 metadata
from langchain_qdrant import QdrantVectorStore  # LangChain 对 Qdrant 的向量库封装
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归文本切分器
from qdrant_client import QdrantClient, models  # QdrantClient 用于按 document_id 删除向量

from model.factory import embed_model  # embedding 模型，用于把文本分片转成向量
from rag.document_parser import DocumentParser  # 通用文档解析器，负责识别、切分和 FAQ 抽取
from rag.file_processors import FileProcessorFactory  # 文件处理器工厂，根据类型选择具体读取策略
from utils.config_handler import qdrant_conf  # 读取 config/qdrant.yml 中的向量库配置
from utils.file_handler import (  # 文件处理工具
    pdf_loader,  # PDF 加载器
    txt_loader,  # TXT 加载器
)
from utils.logger_handler import logger  # 项目统一日志
from utils.qdrant_options import (
    get_qdrant_client_options,  # 读取 Qdrant 连接参数，如 url/host/port/grpc_port
    get_qdrant_distance,  # 读取向量距离算法，如 COSINE
    normalize_qdrant_collection_name,
)


class VectorStoreService:
    """Qdrant 向量库服务。

    这个类主要负责三件事：

    1. 初始化 QdrantVectorStore
       - 连接 Qdrant
       - 指定 collection
       - 指定 embedding 模型
       - 指定向量距离算法

    2. 把单个 documents 表文件写入向量库
       - 读取文件为 Document
       - 按用户确认的策略切分 Document
       - 调用 embedding
       - 写入 Qdrant

    3. 提供 retriever 给 RAG 使用
       - 用户问题会先转成向量
       - Qdrant 按相似度召回 topK 文本分片

    注意：
    - Qdrant 里不是保存“完整文件”，而是保存“文本分片 + 向量 + metadata”。
    - MD5 只能避免重复加载同一份文件，不能自动删除旧版本分片。
    """

    def __init__(self, *, force_recreate: bool | None = None, collection_name: str | None = None):
        """初始化向量库连接和文本切分器。

        这里只准备好 QdrantVectorStore 和 splitter。
        真正写入向量库由 index_file() 完成。
        """

        recreate_collection = qdrant_conf.get("force_recreate", False) if force_recreate is None else force_recreate
        self.collection_name = normalize_qdrant_collection_name(collection_name)

        self.vector_store = QdrantVectorStore.construct_instance(
            embedding=embed_model,  # 文本转向量时使用的 embedding 模型
            collection_name=self.collection_name,  # Qdrant collection 名称
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
    def recreate_collection_service(cls, collection_name: str | None = None) -> "VectorStoreService":
        """删除并重建当前 Qdrant collection，然后返回新的向量库服务实例。

        这个方法只给“全量重建索引”使用。
        它可以清理历史遗留的旧 points，尤其是没有 document_id 的旧数据。
        """

        return cls(force_recreate=True, collection_name=collection_name)

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

        filters 的结构来自 QueryAnalysis，例如：

            {
                "unit_type": ["qa", "numbered"],
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

    def list_documents_by_metadata(
            self,
            field_name: str,
            field_value: str,
            *,
            limit: int = 1000,
    ) -> list[Document]:
        """从当前 collection 按 metadata 字段读取文档。

        这个方法用于“上传预览”等不需要相似度搜索的场景：
        - 前端要按 batch_id 看本次上传切出的所有片段；
        - 服务端要把临时 collection 的片段复制到正式 collection。
        """

        return self.scroll_documents_by_metadata(
            field_name,
            field_value,
            collection_name=self.collection_name,
            limit=limit,
        )

    def delete_by_metadata(self, field_name: str, field_value: str) -> None:
        """从当前 collection 按 metadata 字段删除向量点。"""

        self.delete_vectors_by_metadata(field_name, field_value, collection_name=self.collection_name)

    def copy_points_by_metadata_to(
            self,
            target_service: "VectorStoreService",
            field_name: str,
            field_value: str,
            *,
            metadata_updates: dict[str, Any] | None = None,
            limit: int = 5000,
    ) -> int:
        """把当前 collection 中命中 metadata 的向量点复制到目标 collection。

        复制时保留已有向量，不重新调用 embedding。
        metadata_updates 用于把临时点标记成正式发布点，例如 status=published。
        """

        return self.copy_points_by_metadata(
            source_collection_name=self.collection_name,
            target_collection_name=target_service.collection_name,
            field_name=field_name,
            field_value=field_value,
            metadata_updates=metadata_updates,
            limit=limit,
        )

    def update_metadata_by_metadata(
            self,
            field_name: str,
            field_value: str,
            *,
            metadata_updates: dict[str, Any],
            limit: int = 5000,
    ) -> int:
        """更新当前 collection 中命中 metadata 的点 payload。"""

        return self.copy_points_by_metadata(
            source_collection_name=self.collection_name,
            target_collection_name=self.collection_name,
            field_name=field_name,
            field_value=field_value,
            metadata_updates=metadata_updates,
            limit=limit,
        )

    def preview_file(
            self,
            *,
            filename: str,
            file_path: str,
            sample_limit: int = 5000,
    ) -> dict:
        """读取文件样本文本并识别文档类型。

        这个方法用于上传预览接口，不写 MySQL，也不写 Qdrant。
        """

        documents = self.get_file_documents(file_path)
        sample_text = "\n\n".join(document.page_content for document in documents)[:sample_limit]
        outline = documents[0].metadata.get("_pdf_outline") if documents else None
        detection = self.document_parser.detect_document_type(filename, sample_text, outline=outline)
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

        try:
            return FileProcessorFactory.load_documents(read_path)
        except ValueError:
            return []

    @staticmethod
    def read_pdf_outline(read_path: str) -> list[dict]:
        """读取 PDF 书签目录，返回 level/title/page 结构。"""

        processor = FileProcessorFactory.get_processor("pdf")
        if hasattr(processor, "read_pdf_outline"):
            return processor.read_pdf_outline(read_path)
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
    ) -> list[Document]:
        """把原始 Document 切分成可写入 Qdrant 的文档。

        segments / qa_items 只在内存中用于构造 Qdrant payload。
        MySQL 不再保存知识正文或 FAQ 答案。
        """

        segments, qa_items = self.document_parser.build_segments_and_qas(
            document_id=document_id,
            documents=documents,
            document_type=document_type,
            split_strategy=split_strategy,
        )
        qa_by_segment_id = {item.segment_id: item for item in qa_items}
        index_documents: list[Document] = []

        for segment in segments:
            qa_item = qa_by_segment_id.get(segment.segment_id)
            metadata = {
                "document_id": document_id,
                "segment_id": segment.segment_id,
                "chunk_id": segment.segment_id,
                "content_type": "qa" if qa_item else "segment",
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
                "question_no": qa_item.question_no if qa_item else segment.metadata.get("question_no"),
                "qa_id": qa_item.qa_id if qa_item else None,
                "question": qa_item.question if qa_item else None,
                "category": qa_item.category if qa_item else segment.heading_path,
            }
            metadata.update(segment.metadata)
            if qa_item:
                metadata.update(qa_item.metadata)

            index_documents.append(Document(page_content=segment.content, metadata=metadata))

        return index_documents

    @staticmethod
    def delete_document_vectors(document_id: str, collection_name: str | None = None) -> None:
        """按 document_id 删除 Qdrant 中属于某个文件的所有向量。"""

        client = QdrantClient(**get_qdrant_client_options())
        client.delete(
            collection_name=normalize_qdrant_collection_name(collection_name),
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

    @staticmethod
    def delete_vectors_by_metadata(field_name: str, field_value: str, collection_name: str | None = None) -> None:
        """按 metadata 字段删除 Qdrant 向量。

        训练资料没有写入 documents 表，所以没有 document_id。
        这里用 batch_id 等业务字段删除对应 points。
        """

        client = QdrantClient(**get_qdrant_client_options())
        normalized_collection_name = normalize_qdrant_collection_name(collection_name)
        if not client.collection_exists(normalized_collection_name):
            logger.info("[Qdrant] collection 不存在，跳过向量删除 collection=%s", normalized_collection_name)
            return
        client.delete(
            collection_name=normalized_collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=f"metadata.{field_name}",
                            match=models.MatchValue(value=field_value),
                        )
                    ]
                )
            ),
            wait=True,
        )

    @staticmethod
    def scroll_documents_by_metadata(
            field_name: str,
            field_value: str,
            *,
            collection_name: str | None = None,
            limit: int = 1000,
    ) -> list[Document]:
        """按 metadata 字段从 Qdrant 滚动读取 Document。

        LangChain QdrantVectorStore 默认把正文放在 payload.page_content，
        把元数据放在 payload.metadata。这里按这个结构还原为 Document。
        """

        client = QdrantClient(**get_qdrant_client_options())
        normalized_collection_name = normalize_qdrant_collection_name(collection_name)
        if not client.collection_exists(normalized_collection_name):
            logger.info("[Qdrant] collection 不存在，返回空文档列表 collection=%s", normalized_collection_name)
            return []

        documents: list[Document] = []
        next_page_offset = None
        safe_limit = max(1, limit)
        while len(documents) < safe_limit:
            page_limit = min(256, safe_limit - len(documents))
            points, next_page_offset = client.scroll(
                collection_name=normalized_collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=f"metadata.{field_name}",
                            match=models.MatchValue(value=field_value),
                        )
                    ]
                ),
                limit=page_limit,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                raw_metadata = payload.get("metadata") if isinstance(payload, dict) else {}
                metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
                content = ""
                if isinstance(payload, dict):
                    content = str(payload.get("page_content") or payload.get("content") or payload.get("text") or "")
                documents.append(Document(page_content=content, metadata=metadata))
            if next_page_offset is None:
                break

        return documents

    @staticmethod
    def copy_points_by_metadata(
            *,
            source_collection_name: str | None,
            target_collection_name: str | None,
            field_name: str,
            field_value: str,
            metadata_updates: dict[str, Any] | None = None,
            limit: int = 5000,
    ) -> int:
        """按 metadata 复制或更新 Qdrant 点。

        source 和 target 不同时是跨 collection 复制；
        source 和 target 相同时是原地更新 payload。
        """

        client = QdrantClient(**get_qdrant_client_options())
        source_name = normalize_qdrant_collection_name(source_collection_name)
        target_name = normalize_qdrant_collection_name(target_collection_name)
        if not client.collection_exists(source_name):
            logger.info("[Qdrant] 源 collection 不存在，跳过点复制 collection=%s", source_name)
            return 0
        if not client.collection_exists(target_name):
            logger.info("[Qdrant] 目标 collection 不存在，跳过点复制 collection=%s", target_name)
            return 0

        copied_count = 0
        next_page_offset = None
        safe_limit = max(1, limit)
        updates = metadata_updates or {}
        while copied_count < safe_limit:
            page_limit = min(256, safe_limit - copied_count)
            points, next_page_offset = client.scroll(
                collection_name=source_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=f"metadata.{field_name}",
                            match=models.MatchValue(value=field_value),
                        )
                    ]
                ),
                limit=page_limit,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=True,
            )
            upsert_points: list[models.PointStruct] = []
            for point in points:
                if point.vector is None:
                    continue
                payload = dict(point.payload or {})
                raw_metadata = payload.get("metadata")
                metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
                metadata.update(updates)
                payload["metadata"] = metadata
                upsert_points.append(models.PointStruct(id=point.id, vector=point.vector, payload=payload))

            if upsert_points:
                client.upsert(collection_name=target_name, points=upsert_points, wait=True)
                copied_count += len(upsert_points)

            if next_page_offset is None:
                break

        return copied_count

    @staticmethod
    def list_collections() -> list[str]:
        client = QdrantClient(**get_qdrant_client_options())
        return [collection.name for collection in client.get_collections().collections]

    def index_file(
            self,
            document: dict,
            *,
            document_type: str | None = None,
            split_strategy: str | None = None,
    ) -> int:
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

        document_type = document_type or document.get("document_type") or "text"
        split_strategy = split_strategy or document.get("split_strategy") or "recursive"

        index_documents = self.build_index_documents(
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

        self.delete_document_vectors(document_id, collection_name=self.collection_name)
        self.vector_store.add_documents(index_documents)

        return len(index_documents)
