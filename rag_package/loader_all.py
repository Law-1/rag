import os
import logging
from typing import List, Dict, Callable, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
    UnstructuredXMLLoader,
    UnstructuredRSTLoader,
)

logger = logging.getLogger(__name__)

# 后缀 → 加载器类映射（可按需增删）
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".csv": CSVLoader,
    ".json": JSONLoader,
    ".xml": UnstructuredXMLLoader,
    ".rst": UnstructuredRSTLoader,
    # 添加更多需要时，只需在此映射表中注册
}

DEFAULT_SUPPORTED_EXTS = set(LOADER_MAP.keys())


def load_documents_from_directory(
    input_dir: str = "./docs",
    *,
    recursive: bool = True,
    required_exts: Optional[List[str]] = None,
    exclude_hidden: bool = True,
    raise_on_error: bool = False,
    post_processors: Optional[List[Callable[[List[Document]], List[Document]]]] = None,
    log_level: int = logging.INFO,
) -> List[Document]:
    """
    从目录加载所有支持的文档，返回 LangChain Document 列表。

    Args:
        input_dir: 文档目录
        recursive: 是否递归子目录
        required_exts: 允许的后缀（如 ['.pdf', '.docx']），默认使用所有支持的格式
        exclude_hidden: 是否排除隐藏文件
        raise_on_error: 单个文件失败时是否抛出异常
        post_processors: 文档列表后处理函数
        log_level: 日志级别

    Returns:
        List[Document] 对象，与你的其余代码完全兼容
    """
    logger.setLevel(log_level)

    dir_path = Path(input_dir).resolve()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"目录不存在: {dir_path}")

    # 确定后缀白名单
    if required_exts is None:
        valid_exts = DEFAULT_SUPPORTED_EXTS
    else:
        valid_exts = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                      for ext in required_exts}

    all_documents = []

    # 遍历文件
    pattern = "**/*" if recursive else "*"
    for file_path in dir_path.glob(pattern):
        # 只保留文件，跳过文件夹
        if not file_path.is_file():
            continue
        # 忽略隐藏文件
        if exclude_hidden and file_path.name.startswith('.'):
            continue
        # 只保留允许的后缀 .suffix 获取文件后缀；.lower() 统一小写
        ext = file_path.suffix.lower()
        if ext not in valid_exts:
            continue
        # 存在对应文件加载器才处理
        loader_class = LOADER_MAP.get(ext)
        if not loader_class:
            continue

        try:
            loader = loader_class(str(file_path))
            docs = loader.load()
            # 统一补充文件元数据
            for doc in docs:
                doc.metadata.setdefault("source", str(file_path))
                doc.metadata.setdefault("file_name", file_path.name)
                doc.metadata.setdefault("file_type", ext)
            all_documents.extend(docs)
            logger.info(f"成功加载: {file_path} ({len(docs)} 页/段)")
        except Exception as e:
            logger.error(f"加载失败 {file_path}: {e}")
            if raise_on_error:
                raise

    # 后处理
    if post_processors:
        for processor in post_processors:
            try:
                all_documents = processor(all_documents)
            except Exception as e:
                logger.error(f"后处理失败: {e}")
                if raise_on_error:
                    raise

    # 统计
    ext_stats = {}
    for doc in all_documents:
        ft = doc.metadata.get("file_type", "unknown")
        ext_stats[ft] = ext_stats.get(ft, 0) + 1

    logger.info(f"加载完成: 总计 {len(all_documents)} 条文档 | 类型分布: {ext_stats}")
    return all_documents