"""文件处理器工厂。

这里使用工厂方法模式，根据文件后缀选择 TXT/PDF/DOCX 处理器。
调用方不需要关心具体文件格式，只调用 `FileProcessorFactory.load_documents()`。
"""

from pathlib import Path

from core.rag.file_processors.base import BaseFileProcessor
from core.rag.file_processors.docx_processor import DocxFileProcessor
from core.rag.file_processors.pdf_processor import PdfFileProcessor
from core.rag.file_processors.txt_processor import TxtFileProcessor


class FileProcessorFactory:
    """知识库文件处理器工厂。

    工厂只负责根据文件类型选择处理器，具体读取逻辑交给各处理器策略实现。
    """

    _processors: list[BaseFileProcessor] = []

    @classmethod
    def ensure_default_processors(cls) -> None:
        """注册系统默认文件处理器。"""

        if cls._processors:
            return
        cls.register(TxtFileProcessor())
        cls.register(PdfFileProcessor())
        cls.register(DocxFileProcessor())

    @classmethod
    def register(cls, processor: BaseFileProcessor) -> None:
        """注册一个文件处理器。"""

        processor_class = processor.__class__
        if any(isinstance(existing, processor_class) for existing in cls._processors):
            return
        cls._processors.append(processor)

    @classmethod
    def get_processor(cls, file_type: str) -> BaseFileProcessor:
        """按文件类型获取对应处理器。"""

        cls.ensure_default_processors()
        normalized_type = BaseFileProcessor.normalize_file_type(file_type)
        for processor in cls._processors:
            if processor.support_file_type(normalized_type):
                return processor
        raise ValueError(f"不支持的知识库文件类型：{normalized_type}")

    @classmethod
    def get_processor_for_path(cls, file_path: str) -> BaseFileProcessor:
        """按文件路径后缀获取对应处理器。"""

        file_type = BaseFileProcessor.normalize_file_type(Path(file_path).suffix)
        return cls.get_processor(file_type)

    @classmethod
    def load_documents(cls, file_path: str):
        """读取文件并返回统一的 Document 列表。"""

        return cls.get_processor_for_path(file_path).load_documents(file_path)
