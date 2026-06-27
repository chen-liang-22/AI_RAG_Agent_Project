"""文件读取和文件指纹工具。

这里保留少量通用文件函数：
- 计算 MD5，用于上传去重；
- 按后缀扫描 data 目录；
- 兼容旧代码的 PDF/TXT loader。

新业务上传文件长期保存统一走 MinIO，不再依赖本地 uploads 目录。
"""

import os
import hashlib
from core.utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(filepath: str) -> str | None:
    """计算文件 MD5 十六进制字符串。

    MD5 在本项目只用于“内容是否重复”的工程判断，不用于安全加密。
    """

    if not os.path.exists(filepath):
        logger.error(f"[MD5计算] 文件{filepath}不存在")
        return

    if not os.path.isfile(filepath):
        logger.error(f"[MD5计算] 路径{filepath}不是文件")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096       # 4KB分片，避免文件过大爆内存
    try:
        with open(filepath, "rb") as f:     # 必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            """
            chunk = f.read(chunk_size)
            while chunk:
                
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[MD5计算] 计算文件{filepath}失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]) -> tuple[str, ...]:
    """返回目录下后缀在 allowed_types 范围内的文件路径。"""

    files = []

    if not os.path.isdir(path):
        logger.error(f"[文件列表] {path}不是文件夹")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """读取 PDF 文件为 LangChain Document 列表。"""

    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    """读取 TXT 文件为 LangChain Document 列表。"""

    return TextLoader(filepath, encoding="utf-8").load()
