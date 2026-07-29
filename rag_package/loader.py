import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

def load_from_directory(directory: str = "./file") -> List[Document]:
    """从指定目录加载所有 PDF，返回合并文档列表"""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"目录不存在: {directory}")

    file_list = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.lower().endswith(".pdf")]
    if not file_list:
        raise ValueError("未找到任何 PDF 文件，请检查路径")

    all_docs = []
    for path in file_list:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"成功加载：{path}，页数：{len(docs)}")
        except Exception as e:
            print(f"加载失败 {path}, err: {str(e)}")
    return all_docs